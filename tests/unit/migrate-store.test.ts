import { afterEach, describe, it, expect, vi } from "vitest";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { runMigrateStore } from "../../src/migrate-store.js";
import { openSqliteStore } from "../../src/store/sqlite-store.js";
import type { Config } from "../../src/config.js";

const tmpDirs: string[] = [];
async function makeTmpDir(): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), "oracle-mig-"));
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

function vectorLine(repo: string, filePath: string, embedding: number[], fileHash = "h1") {
  return JSON.stringify({
    embedding,
    doc: {
      pageContent: `${repo}/${filePath}`,
      metadata: { repo, filePath, fileHash },
    },
  });
}

describe("runMigrateStore", () => {
  it("converts a JSONL with meta line into a SQLite store and moves the source to .bak", async () => {
    const dir = await makeTmpDir();
    const jsonlPath = join(dir, "embeddings.jsonl");
    const metaLine = JSON.stringify({
      type: "meta",
      embeddingProvider: "openai",
      embeddingModel: "text-embedding-3-small",
      dimension: 3,
      embeddedAt: new Date().toISOString(),
      count: 2,
    });
    const body = [
      metaLine,
      vectorLine("auth", "auth/a.ts", normalized([1, 0, 0])),
      vectorLine("billing", "billing/a.ts", normalized([0, 1, 0])),
    ].join("\n");
    await writeFile(jsonlPath, body + "\n", "utf8");

    const config = testConfig(dir);
    await runMigrateStore(config);

    expect(existsSync(jsonlPath)).toBe(false);
    expect(existsSync(join(dir, ".embeddings.jsonl.bak"))).toBe(true);

    const store = openSqliteStore(config);
    try {
      expect(store.count()).toBe(2);
      const meta = store.getMeta();
      expect(meta?.embeddingProvider).toBe("openai");
      expect(meta?.embeddingModel).toBe("text-embedding-3-small");
      expect(meta?.dimension).toBe(3);
      const results = store.similaritySearch(normalized([1, 0, 0]), 2);
      expect(results[0].metadata.filePath).toBe("auth/a.ts");
    } finally {
      store.close();
    }
  });

  it("is a no-op when no embeddings.jsonl exists", async () => {
    const dir = await makeTmpDir();
    await runMigrateStore(testConfig(dir));
    expect(existsSync(join(dir, "store.db"))).toBe(false);
  });

  it("refuses to overwrite an existing non-empty store", async () => {
    const dir = await makeTmpDir();
    // Pre-seed SQLite store.
    const config = testConfig(dir);
    const store = openSqliteStore(config);
    store.initializeSchema({
      embeddingProvider: "openai",
      embeddingModel: "text-embedding-3-small",
      dimension: 3,
    });
    store.insertBatch([
      {
        embedding: normalized([1, 0, 0]),
        pageContent: "x",
        metadata: { repo: "r", filePath: "r/x.ts" },
      },
    ]);
    store.close();

    const jsonlPath = join(dir, "embeddings.jsonl");
    await writeFile(
      jsonlPath,
      [
        JSON.stringify({
          type: "meta",
          embeddingProvider: "openai",
          embeddingModel: "text-embedding-3-small",
          dimension: 3,
        }),
        vectorLine("r", "r/y.ts", normalized([0, 1, 0])),
      ].join("\n") + "\n",
      "utf8",
    );

    await expect(runMigrateStore(config)).rejects.toThrow(/refusing to overwrite/);
  });

  it("refuses to migrate a JSONL with vectors of mixed dimensions", async () => {
    const dir = await makeTmpDir();
    const jsonlPath = join(dir, "embeddings.jsonl");
    await writeFile(
      jsonlPath,
      [
        vectorLine("r", "r/a.ts", normalized([1, 0, 0])),
        vectorLine("r", "r/b.ts", normalized([1, 0, 0, 0])),
      ].join("\n") + "\n",
      "utf8",
    );
    await expect(runMigrateStore(testConfig(dir))).rejects.toThrow(/mixed dimensions/);
  });

  it("refuses to migrate when meta dimension disagrees with file vectors", async () => {
    const dir = await makeTmpDir();
    const jsonlPath = join(dir, "embeddings.jsonl");
    const metaLine = JSON.stringify({
      type: "meta",
      embeddingProvider: "openai",
      embeddingModel: "text-embedding-3-small",
      dimension: 1536,
    });
    await writeFile(
      jsonlPath,
      [metaLine, vectorLine("r", "r/a.ts", normalized([1, 0, 0]))].join("\n") + "\n",
      "utf8",
    );
    await expect(runMigrateStore(testConfig(dir))).rejects.toThrow(
      /meta says dimension 1536 but vectors are dimension 3/,
    );
  });

  it("falls back to current config when legacy JSONL lacks fingerprint meta", async () => {
    const dir = await makeTmpDir();
    const jsonlPath = join(dir, "embeddings.jsonl");
    await writeFile(
      jsonlPath,
      [vectorLine("r", "r/x.ts", normalized([1, 0, 0]))].join("\n") + "\n",
      "utf8",
    );
    const config = testConfig(dir);
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      await runMigrateStore(config);
    } finally {
      warnSpy.mockRestore();
    }

    const store = openSqliteStore(config);
    try {
      expect(store.count()).toBe(1);
      const meta = store.getMeta();
      expect(meta?.embeddingProvider).toBe("openai");
      expect(meta?.dimension).toBe(3);
    } finally {
      store.close();
    }
  });
});

