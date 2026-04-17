import { afterEach, describe, it, expect } from "vitest";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import {
  IndexFingerprintError,
  openSqliteStore,
  type StoredEntry,
} from "../../src/store/sqlite-store.js";
import type { Config } from "../../src/config.js";

const tmpDirs: string[] = [];
async function makeTmpDir(): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), "oracle-ss-"));
  tmpDirs.push(dir);
  return dir;
}

function testConfig(dir: string, overrides: Partial<Config> = {}): Config {
  return {
    scanRoot: "/tmp",
    dataDir: dir,
    embeddingProvider: "openai",
    llmProvider: "auto",
    ollamaBaseUrl: "http://localhost:11434/v1",
    embeddingModel: "text-embedding-3-small",
    llmModel: "test",
    vectorStoreType: "directory",
    ...overrides,
  };
}

function entry(
  repo: string,
  filePath: string,
  embedding: number[],
  pageContent = "x",
  fileHash = "h1",
): StoredEntry {
  return {
    embedding,
    pageContent,
    metadata: { repo, filePath, fileHash },
  };
}

function normalized(values: number[]): number[] {
  const norm = Math.sqrt(values.reduce((s, v) => s + v * v, 0)) || 1;
  return values.map((v) => v / norm);
}

afterEach(async () => {
  while (tmpDirs.length > 0) {
    const dir = tmpDirs.pop();
    if (dir) await rm(dir, { recursive: true, force: true });
  }
});

describe("openSqliteStore basics", () => {
  it("starts empty and reports no meta", async () => {
    const dir = await makeTmpDir();
    const store = openSqliteStore(testConfig(dir));
    expect(store.getMeta()).toBeNull();
    expect(store.count()).toBe(0);
    expect(store.listRepos()).toEqual([]);
    store.close();
  });

  it("initializeSchema writes provider/model/dimension", async () => {
    const dir = await makeTmpDir();
    const store = openSqliteStore(testConfig(dir));
    store.initializeSchema({
      embeddingProvider: "openai",
      embeddingModel: "text-embedding-3-small",
      dimension: 4,
    });
    const meta = store.getMeta();
    expect(meta?.embeddingProvider).toBe("openai");
    expect(meta?.embeddingModel).toBe("text-embedding-3-small");
    expect(meta?.dimension).toBe(4);
    expect(meta?.schemaVersion).toBe("1");
    store.close();
  });

  it("initializeSchema is idempotent for matching meta and refuses mismatched ones", async () => {
    const dir = await makeTmpDir();
    const store = openSqliteStore(testConfig(dir));
    store.initializeSchema({ embeddingProvider: "openai", embeddingModel: "m", dimension: 4 });
    // Repeat with identical meta: no throw.
    store.initializeSchema({ embeddingProvider: "openai", embeddingModel: "m", dimension: 4 });
    // Different dimension: throws.
    expect(() =>
      store.initializeSchema({ embeddingProvider: "openai", embeddingModel: "m", dimension: 8 }),
    ).toThrow(IndexFingerprintError);
    // Different provider: throws.
    expect(() =>
      store.initializeSchema({ embeddingProvider: "ollama", embeddingModel: "m", dimension: 4 }),
    ).toThrow(IndexFingerprintError);
    // Different model: throws.
    expect(() =>
      store.initializeSchema({ embeddingProvider: "openai", embeddingModel: "n", dimension: 4 }),
    ).toThrow(IndexFingerprintError);
    store.close();
  });

  it("assertCompatibleWithConfig: empty store passes regardless of config", async () => {
    const dir = await makeTmpDir();
    const store = openSqliteStore(testConfig(dir));
    expect(() =>
      store.assertCompatibleWithConfig(
        testConfig(dir, { embeddingProvider: "ollama", embeddingModel: "other" }),
      ),
    ).not.toThrow();
    store.close();
  });

  it("assertCompatibleWithConfig throws on provider or model drift", async () => {
    const dir = await makeTmpDir();
    const store = openSqliteStore(testConfig(dir));
    store.initializeSchema({ embeddingProvider: "openai", embeddingModel: "m1", dimension: 4 });

    expect(() =>
      store.assertCompatibleWithConfig(testConfig(dir, { embeddingProvider: "ollama" })),
    ).toThrow(/provider "openai"/);

    expect(() =>
      store.assertCompatibleWithConfig(testConfig(dir, { embeddingModel: "m2" })),
    ).toThrow(/model "m1"/);
    store.close();
  });
});

describe("CRUD + similarity", () => {
  it("insertBatch + similaritySearch returns nearest neighbours first", async () => {
    const dir = await makeTmpDir();
    const store = openSqliteStore(testConfig(dir));
    store.initializeSchema({ embeddingProvider: "openai", embeddingModel: "m", dimension: 3 });
    store.insertBatch([
      entry("r", "r/a.ts", normalized([1, 0, 0])),
      entry("r", "r/b.ts", normalized([0, 1, 0])),
      entry("r", "r/c.ts", normalized([0.9, 0.1, 0])),
    ]);
    expect(store.count()).toBe(3);

    const results = store.similaritySearch(normalized([1, 0, 0]), 2);
    expect(results).toHaveLength(2);
    expect(results[0].metadata.filePath).toBe("r/a.ts");
    expect(results[1].metadata.filePath).toBe("r/c.ts");
    expect(results[0].distance).toBeLessThan(results[1].distance);
    store.close();
  });

  it("similaritySearch filters by repo", async () => {
    const dir = await makeTmpDir();
    const store = openSqliteStore(testConfig(dir));
    store.initializeSchema({ embeddingProvider: "openai", embeddingModel: "m", dimension: 3 });
    store.insertBatch([
      entry("auth", "auth/a.ts", normalized([1, 0, 0.1])),
      entry("billing", "billing/a.ts", normalized([1, 0, 0.05])),
      entry("auth", "auth/b.ts", normalized([0.9, 0.1, 0])),
    ]);
    const results = store.similaritySearch(normalized([1, 0, 0]), 5, { repo: "auth" });
    expect(results).toHaveLength(2);
    for (const r of results) expect(r.metadata.repo).toBe("auth");
    store.close();
  });

  it("upsertFile atomically replaces per-file chunks", async () => {
    const dir = await makeTmpDir();
    const store = openSqliteStore(testConfig(dir));
    store.initializeSchema({ embeddingProvider: "openai", embeddingModel: "m", dimension: 3 });
    store.insertBatch([
      entry("r", "r/a.ts", normalized([1, 0, 0]), "v1", "h1"),
      entry("r", "r/a.ts", normalized([0.9, 0.1, 0]), "v1b", "h1"),
      entry("r", "r/b.ts", normalized([0, 1, 0]), "u", "h-other"),
    ]);
    expect(store.count()).toBe(3);
    const result = store.upsertFile("r", "r/a.ts", "h2", [
      entry("r", "r/a.ts", normalized([0, 0, 1]), "v2", "h2"),
    ]);
    expect(result.added).toBe(1);
    expect(result.removed).toBe(2);
    expect(store.count()).toBe(2);

    const results = store.similaritySearch(normalized([0, 0, 1]), 1);
    expect(results[0].pageContent).toBe("v2");
    store.close();
  });

  it("deleteByFile and deleteByRepo remove both docs and vectors", async () => {
    const dir = await makeTmpDir();
    const store = openSqliteStore(testConfig(dir));
    store.initializeSchema({ embeddingProvider: "openai", embeddingModel: "m", dimension: 3 });
    store.insertBatch([
      entry("auth", "auth/a.ts", normalized([1, 0, 0])),
      entry("auth", "auth/b.ts", normalized([0, 1, 0])),
      entry("billing", "billing/a.ts", normalized([0, 0, 1])),
    ]);

    expect(store.deleteByFile("auth", "auth/a.ts")).toBe(1);
    expect(store.count()).toBe(2);

    expect(store.deleteByRepo("billing")).toBe(1);
    expect(store.count()).toBe(1);

    // After deletions, similaritySearch cannot return the dropped rows.
    const results = store.similaritySearch(normalized([1, 0, 0]), 10);
    const paths = results.map((r) => r.metadata.filePath);
    expect(paths).not.toContain("auth/a.ts");
    expect(paths).not.toContain("billing/a.ts");
    store.close();
  });

  it("fileSignatures returns the latest per-file hash", async () => {
    const dir = await makeTmpDir();
    const store = openSqliteStore(testConfig(dir));
    store.initializeSchema({ embeddingProvider: "openai", embeddingModel: "m", dimension: 3 });
    store.insertBatch([
      entry("r", "r/a.ts", normalized([1, 0, 0]), "x", "hash-a"),
      entry("r", "r/a.ts", normalized([0.9, 0, 0.1]), "x2", "hash-a"),
      entry("r", "r/b.ts", normalized([0, 1, 0]), "y", "hash-b"),
    ]);
    const sigs = store.fileSignatures();
    expect(sigs.size).toBe(2);
    expect(sigs.get("r::r/a.ts")?.fileHash).toBe("hash-a");
    expect(sigs.get("r::r/b.ts")?.fileHash).toBe("hash-b");
    store.close();
  });

  it("write epoch advances on every mutation", async () => {
    const dir = await makeTmpDir();
    const store = openSqliteStore(testConfig(dir));
    store.initializeSchema({ embeddingProvider: "openai", embeddingModel: "m", dimension: 3 });
    expect(store.getWriteEpoch()).toBe(0);
    store.insertBatch([entry("r", "r/a.ts", normalized([1, 0, 0]))]);
    const e1 = store.getWriteEpoch();
    expect(e1).toBeGreaterThan(0);
    store.upsertFile("r", "r/a.ts", "h2", [
      entry("r", "r/a.ts", normalized([0, 1, 0]), "y", "h2"),
    ]);
    expect(store.getWriteEpoch()).toBeGreaterThan(e1);
    store.close();
  });
});

describe("initializeSchema contention", () => {
  it("two handles racing the first initialize converge without corrupting the dim lock", async () => {
    const dir = await makeTmpDir();
    const s1 = openSqliteStore(testConfig(dir));
    const s2 = openSqliteStore(testConfig(dir));
    try {
      // Both handles start with no meta and would try to create the vec0
      // table. With IMMEDIATE + re-check under lock, one wins, the other
      // sees matching meta and is a no-op.
      s1.initializeSchema({ embeddingProvider: "openai", embeddingModel: "m", dimension: 4 });
      s2.initializeSchema({ embeddingProvider: "openai", embeddingModel: "m", dimension: 4 });
      expect(s1.getMeta()?.dimension).toBe(4);
      expect(s2.getMeta()?.dimension).toBe(4);
      // A racing call with a DIFFERENT dim throws once the store is populated.
      expect(() =>
        s2.initializeSchema({ embeddingProvider: "openai", embeddingModel: "m", dimension: 8 }),
      ).toThrow(IndexFingerprintError);
    } finally {
      s1.close();
      s2.close();
    }
  });
});

describe("concurrency (WAL)", () => {
  it("a separate reader sees writes from another connection without reopening", async () => {
    const dir = await makeTmpDir();
    const writer = openSqliteStore(testConfig(dir));
    writer.initializeSchema({ embeddingProvider: "openai", embeddingModel: "m", dimension: 3 });
    writer.insertBatch([entry("r", "r/a.ts", normalized([1, 0, 0]))]);

    const reader = openSqliteStore(testConfig(dir));
    expect(reader.count()).toBe(1);

    writer.upsertFile("r", "r/b.ts", "h", [
      entry("r", "r/b.ts", normalized([0, 1, 0]), "v", "h"),
    ]);

    // The reader sees the new row via a fresh query on the same handle.
    expect(reader.count()).toBe(2);
    const results = reader.similaritySearch(normalized([0, 1, 0]), 1);
    expect(results[0].metadata.filePath).toBe("r/b.ts");

    writer.deleteByFile("r", "r/a.ts");
    expect(reader.count()).toBe(1);

    reader.close();
    writer.close();
  });

  it("a separate OS process sees writes from the current process (file-level WAL)", async () => {
    const dir = await makeTmpDir();
    const writer = openSqliteStore(testConfig(dir));
    writer.initializeSchema({ embeddingProvider: "openai", embeddingModel: "m", dimension: 3 });
    writer.insertBatch([
      entry("repoA", "a.ts", normalized([1, 0, 0])),
      entry("repoB", "b.ts", normalized([0, 1, 0])),
    ]);
    writer.close();

    // Spawn an independent Node process that opens the same file and reports
    // the row count + repo list. This proves the "MCP server sees watch
    // writes" claim across process boundaries, not just connections.
    const scriptPath = join(dir, "reader.cjs");
    await writeFile(
      scriptPath,
      `
const Database = require(${JSON.stringify(require.resolve("better-sqlite3"))});
const sqliteVec = require(${JSON.stringify(require.resolve("sqlite-vec"))});
const db = new Database(${JSON.stringify(join(dir, "store.db"))}, { readonly: true });
db.pragma("journal_mode = WAL");
sqliteVec.load(db);
const c = db.prepare("SELECT COUNT(*) AS c FROM docs").get().c;
const repos = db.prepare("SELECT repo, COUNT(*) AS n FROM docs GROUP BY repo ORDER BY repo").all();
console.log(JSON.stringify({ count: c, repos }));
db.close();
      `,
      "utf8",
    );

    const writerAgain = openSqliteStore(testConfig(dir));
    writerAgain.upsertFile("repoA", "a2.ts", "h", [
      entry("repoA", "a2.ts", normalized([0, 0, 1]), "new", "h"),
    ]);
    writerAgain.close();

    const result = spawnSync(process.execPath, [scriptPath], { encoding: "utf8" });
    expect(result.status).toBe(0);
    const parsed = JSON.parse(result.stdout.trim()) as {
      count: number;
      repos: Array<{ repo: string; n: number }>;
    };
    expect(parsed.count).toBe(3);
    const byRepo = Object.fromEntries(parsed.repos.map((r) => [r.repo, r.n]));
    expect(byRepo.repoA).toBe(2);
    expect(byRepo.repoB).toBe(1);
  });
});
