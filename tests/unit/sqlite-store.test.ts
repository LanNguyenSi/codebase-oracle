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
import { splitFile } from "../../src/ingest/splitter.js";
import type { ScannedFile } from "../../src/ingest/scanner.js";
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
    maxFileSizeBytes: 500_000,
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

  it("deleteByFile on a never-initialized store is a no-op, not a crash", async () => {
    // Regression test: watch mode's flush() calls store.deleteByFile for
    // delete/too-large/empty events before any file has ever been embedded.
    // If that event is the very first one a fresh watch instance sees, the
    // store has no embedding dimension yet (initializeSchema was never
    // called) and this used to throw IndexFingerprintError instead of
    // returning 0.
    const dir = await makeTmpDir();
    const store = openSqliteStore(testConfig(dir));
    expect(store.getMeta()).toBeNull();
    expect(() => store.deleteByFile("auth", "auth/a.ts")).not.toThrow();
    expect(store.deleteByFile("auth", "auth/a.ts")).toBe(0);
    expect(store.count()).toBe(0);
    store.close();
  });

  it("deleteByRepo on a never-initialized store is a no-op, not a crash", async () => {
    // Regression test (sibling of the deleteByFile guard above): watch mode's
    // flush() calls store.deleteByRepo for a dropped repo BEFORE any file has
    // ever been embedded (watch.ts flush() drains droppedRepos first). If a
    // repo-drop is the very first event a fresh watch instance sees, the store
    // has no embedding dimension yet (initializeSchema was never called) and
    // this used to throw IndexFingerprintError instead of returning 0.
    const dir = await makeTmpDir();
    const store = openSqliteStore(testConfig(dir));
    expect(store.getMeta()).toBeNull();
    expect(() => store.deleteByRepo("auth")).not.toThrow();
    expect(store.deleteByRepo("auth")).toBe(0);
    expect(store.count()).toBe(0);
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
    store.initializeSchema({
      embeddingProvider: "openai",
      embeddingModel: "m",
      dimension: 4,
    });
    // Repeat with identical meta: no throw.
    store.initializeSchema({
      embeddingProvider: "openai",
      embeddingModel: "m",
      dimension: 4,
    });
    // Different dimension: throws.
    expect(() =>
      store.initializeSchema({
        embeddingProvider: "openai",
        embeddingModel: "m",
        dimension: 8,
      }),
    ).toThrow(IndexFingerprintError);
    // Different provider: throws.
    expect(() =>
      store.initializeSchema({
        embeddingProvider: "ollama",
        embeddingModel: "m",
        dimension: 4,
      }),
    ).toThrow(IndexFingerprintError);
    // Different model: throws.
    expect(() =>
      store.initializeSchema({
        embeddingProvider: "openai",
        embeddingModel: "n",
        dimension: 4,
      }),
    ).toThrow(IndexFingerprintError);
    store.close();
  });

  it("assertCompatibleWithConfig: empty store passes regardless of config", async () => {
    const dir = await makeTmpDir();
    const store = openSqliteStore(testConfig(dir));
    expect(() =>
      store.assertCompatibleWithConfig(
        testConfig(dir, {
          embeddingProvider: "ollama",
          embeddingModel: "other",
        }),
      ),
    ).not.toThrow();
    store.close();
  });

  it("assertCompatibleWithConfig throws on provider or model drift", async () => {
    const dir = await makeTmpDir();
    const store = openSqliteStore(testConfig(dir));
    store.initializeSchema({
      embeddingProvider: "openai",
      embeddingModel: "m1",
      dimension: 4,
    });

    expect(() =>
      store.assertCompatibleWithConfig(
        testConfig(dir, { embeddingProvider: "ollama" }),
      ),
    ).toThrow(/provider "openai"/);

    expect(() =>
      store.assertCompatibleWithConfig(
        testConfig(dir, { embeddingModel: "m2" }),
      ),
    ).toThrow(/model "m1"/);
    store.close();
  });
});

describe("CRUD + similarity", () => {
  it("insertBatch + similaritySearch returns nearest neighbours first", async () => {
    const dir = await makeTmpDir();
    const store = openSqliteStore(testConfig(dir));
    store.initializeSchema({
      embeddingProvider: "openai",
      embeddingModel: "m",
      dimension: 3,
    });
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
    store.initializeSchema({
      embeddingProvider: "openai",
      embeddingModel: "m",
      dimension: 3,
    });
    store.insertBatch([
      entry("auth", "auth/a.ts", normalized([1, 0, 0.1])),
      entry("billing", "billing/a.ts", normalized([1, 0, 0.05])),
      entry("auth", "auth/b.ts", normalized([0.9, 0.1, 0])),
    ]);
    const results = store.similaritySearch(normalized([1, 0, 0]), 5, {
      repo: "auth",
    });
    expect(results).toHaveLength(2);
    for (const r of results) expect(r.metadata.repo).toBe("auth");
    store.close();
  });

  it("upsertFile atomically replaces per-file chunks", async () => {
    const dir = await makeTmpDir();
    const store = openSqliteStore(testConfig(dir));
    store.initializeSchema({
      embeddingProvider: "openai",
      embeddingModel: "m",
      dimension: 3,
    });
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
    store.initializeSchema({
      embeddingProvider: "openai",
      embeddingModel: "m",
      dimension: 3,
    });
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

  it("listRepos.lastIndexedAt is null on a fresh store, advances on writes, clears on deleteByRepo", async () => {
    const dir = await makeTmpDir();
    const store = openSqliteStore(testConfig(dir));
    store.initializeSchema({
      embeddingProvider: "openai",
      embeddingModel: "m",
      dimension: 3,
    });

    // No repos yet.
    expect(store.listRepos()).toEqual([]);

    // First write sets lastIndexedAt.
    store.insertBatch([entry("r", "r/a.ts", normalized([1, 0, 0]))]);
    const afterFirst = store.listRepos();
    expect(afterFirst).toHaveLength(1);
    const firstStamp = afterFirst[0].lastIndexedAt;
    expect(firstStamp).toMatch(/^\d{4}-\d{2}-\d{2}T/);

    // Second write advances lastIndexedAt strictly forward.
    await new Promise((r) => setTimeout(r, 10)); // ensure ISO ms diff
    store.upsertFile("r", "r/b.ts", null, [
      {
        embedding: normalized([0, 1, 0]),
        pageContent: "y",
        metadata: { repo: "r", filePath: "r/b.ts" },
      },
    ]);
    const afterSecond = store.listRepos();
    expect(afterSecond[0].lastIndexedAt).not.toBe(firstStamp);
    expect(
      new Date(afterSecond[0].lastIndexedAt!).getTime(),
    ).toBeGreaterThanOrEqual(new Date(firstStamp!).getTime());

    // deleteByFile bumps the timestamp (file removal IS an index update).
    await new Promise((r) => setTimeout(r, 10));
    const beforeDelete = afterSecond[0].lastIndexedAt!;
    store.deleteByFile("r", "r/a.ts");
    const afterDelete = store.listRepos();
    expect(new Date(afterDelete[0].lastIndexedAt!).getTime()).toBeGreaterThan(
      new Date(beforeDelete).getTime(),
    );

    // deleteByRepo drops both docs and the freshness row.
    store.deleteByRepo("r");
    expect(store.listRepos()).toEqual([]);
    store.close();
  });

  it("deleteByFile drops the repo_meta row when the last file of a repo is removed", async () => {
    // Regression for orphan repo_meta rows: pruning the last doc of a repo
    // through deleteByFile (the path the full-reindex command takes) used to
    // leave a stale freshness row behind that nothing ever cleaned up.
    const dir = await makeTmpDir();
    const store = openSqliteStore(testConfig(dir));
    store.initializeSchema({
      embeddingProvider: "openai",
      embeddingModel: "m",
      dimension: 3,
    });
    store.insertBatch([
      entry("r", "r/a.ts", normalized([1, 0, 0])),
      entry("r", "r/b.ts", normalized([0, 1, 0])),
    ]);

    // Two files, one deletion → repo still has a doc, freshness row stays.
    store.deleteByFile("r", "r/a.ts");
    let listed = store.listRepos();
    expect(listed).toHaveLength(1);
    expect(listed[0].repo).toBe("r");
    expect(listed[0].lastIndexedAt).not.toBeNull();

    // Last file gone → repo disappears AND repo_meta row is dropped.
    store.deleteByFile("r", "r/b.ts");
    listed = store.listRepos();
    expect(listed).toEqual([]);
    // Re-inserting the repo after a full prune must produce a fresh, non-null
    // freshness stamp — i.e. there is no orphan row to overwrite.
    store.insertBatch([entry("r", "r/c.ts", normalized([0, 0, 1]))]);
    listed = store.listRepos();
    expect(listed).toHaveLength(1);
    expect(listed[0].lastIndexedAt).toMatch(/^\d{4}-\d{2}-\d{2}T/);
    store.close();
  });

  it("pruneOrphanRepoMeta clears repo_meta rows whose docs no longer exist", async () => {
    // Backfill-cleanup for stores that predate the deleteByFile fix: orphan
    // repo_meta rows should be removable in one sweep at re-index startup.
    const dir = await makeTmpDir();
    const store = openSqliteStore(testConfig(dir));
    store.initializeSchema({
      embeddingProvider: "openai",
      embeddingModel: "m",
      dimension: 3,
    });
    store.insertBatch([
      entry("kept", "kept/a.ts", normalized([1, 0, 0])),
      entry("orphan-a", "orphan-a/x.ts", normalized([0, 1, 0])),
      entry("orphan-b", "orphan-b/y.ts", normalized([0, 0, 1])),
    ]);

    // Simulate the legacy bug: docs+vectors of two repos are gone but their
    // repo_meta rows survived. We can't reach that state through the public
    // API anymore (deleteByFile now drops meta), so insert the situation by
    // raw deletes — this mirrors what a pre-fix store looks like on disk.
    const Database = (await import("better-sqlite3")).default;
    const sqliteVec = await import("sqlite-vec");
    const dbPath = store.dbPath;
    store.close();
    const raw = new Database(dbPath);
    sqliteVec.load(raw);
    raw
      .prepare(
        "DELETE FROM vectors WHERE rowid IN (SELECT rowid FROM docs WHERE repo LIKE 'orphan-%')",
      )
      .run();
    raw.prepare("DELETE FROM docs WHERE repo LIKE 'orphan-%'").run();
    raw.close();

    const store2 = openSqliteStore(testConfig(dir));
    expect(store2.pruneOrphanRepoMeta()).toBe(2);
    expect(store2.pruneOrphanRepoMeta()).toBe(0); // idempotent
    // The kept repo's freshness row stays.
    expect(store2.listRepos().map((r) => r.repo)).toEqual(["kept"]);
    expect(store2.listRepos()[0].lastIndexedAt).not.toBeNull();
    store2.close();
  });

  it("reindex sequence: prune-by-file + touchRepo(liveRepos) does not re-create orphan rows", async () => {
    // End-to-end-ish regression for the index command's flow. Mirrors the
    // post-walk sequence in src/index.ts: prune files no longer on disk
    // through deleteByFile, then touchRepo only for repos that still have
    // at least one live file. A repo whose entire content vanished must NOT
    // get a touchRepo call — otherwise the repo_meta row that deleteByFile
    // just dropped would be re-created as an orphan.
    const dir = await makeTmpDir();
    const store = openSqliteStore(testConfig(dir));
    store.initializeSchema({
      embeddingProvider: "openai",
      embeddingModel: "m",
      dimension: 3,
    });
    store.insertBatch([
      entry("kept", "kept/a.ts", normalized([1, 0, 0])),
      entry("kept", "kept/b.ts", normalized([0, 1, 0])),
      entry("vanished", "vanished/x.ts", normalized([0, 0, 1])),
      entry("vanished", "vanished/y.ts", normalized([1, 1, 0])),
    ]);

    // Simulate the next reindex run finding `kept` intact but `vanished`
    // gone from disk: walkRepo yields only kept's files, so seenKeys does
    // not contain any of vanished's. liveRepos collects only kept.
    const seenKeys = new Set(["kept::kept/a.ts", "kept::kept/b.ts"]);
    const liveRepos = new Set(["kept"]);
    const sigs = store.fileSignatures();
    for (const [key, sig] of sigs) {
      if (seenKeys.has(key)) continue;
      store.deleteByFile(sig.repo, sig.filePath);
    }
    const scannedAt = new Date().toISOString();
    for (const repo of liveRepos) store.touchRepo(repo, scannedAt);

    const listed = store.listRepos();
    expect(listed.map((r) => r.repo)).toEqual(["kept"]);
    expect(listed[0].lastIndexedAt).toBe(scannedAt);
    // The vanished repo must NOT have re-acquired a repo_meta row.
    const orphans = store.pruneOrphanRepoMeta();
    expect(orphans).toBe(0);
    store.close();
  });

  it("touchRepo stamps last_indexed_at without writing docs", async () => {
    // The full-reindex command calls store.touchRepo for every scanned repo
    // so that repos with zero changes (everything reused) still advance
    // their freshness timestamp. Without this, a watcher-less workflow
    // leaves last_indexed_at = null forever.
    const dir = await makeTmpDir();
    const store = openSqliteStore(testConfig(dir));
    store.initializeSchema({
      embeddingProvider: "openai",
      embeddingModel: "m",
      dimension: 3,
    });
    store.insertBatch([entry("r", "r/a.ts", normalized([1, 0, 0]))]);

    const before = store.listRepos()[0].lastIndexedAt!;
    await new Promise((r) => setTimeout(r, 10));
    const stamp = new Date().toISOString();
    store.touchRepo("r", stamp);
    const after = store.listRepos()[0].lastIndexedAt;
    expect(after).toBe(stamp);
    expect(new Date(after!).getTime()).toBeGreaterThan(
      new Date(before).getTime(),
    );
    store.close();
  });

  it("fileSignatures returns the latest per-file hash", async () => {
    const dir = await makeTmpDir();
    const store = openSqliteStore(testConfig(dir));
    store.initializeSchema({
      embeddingProvider: "openai",
      embeddingModel: "m",
      dimension: 3,
    });
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

  it("getFirstChunkByFile returns the first-inserted chunk (lowest rowid) and null for misses", async () => {
    const dir = await makeTmpDir();
    const store = openSqliteStore(testConfig(dir));
    store.initializeSchema({
      embeddingProvider: "openai",
      embeddingModel: "m",
      dimension: 3,
    });
    // Chunks are inserted in file order, so the first insert is the top of
    // the file and must be what getFirstChunkByFile returns.
    store.insertBatch([
      entry("r", "r/a.ts", normalized([1, 0, 0]), "first-chunk", "h"),
      entry("r", "r/a.ts", normalized([0, 1, 0]), "second-chunk", "h"),
      entry("r", "r/b.ts", normalized([0, 0, 1]), "b-chunk", "h"),
    ]);

    const first = store.getFirstChunkByFile("r", "r/a.ts");
    expect(first?.pageContent).toBe("first-chunk");
    expect(first?.metadata.filePath).toBe("r/a.ts");
    expect(first?.metadata.repo).toBe("r");

    // Miss: unknown file and wrong-repo scoping both return null.
    expect(store.getFirstChunkByFile("r", "r/missing.ts")).toBeNull();
    expect(store.getFirstChunkByFile("other", "r/a.ts")).toBeNull();
    store.close();
  });

  it("write epoch advances on every mutation", async () => {
    const dir = await makeTmpDir();
    const store = openSqliteStore(testConfig(dir));
    store.initializeSchema({
      embeddingProvider: "openai",
      embeddingModel: "m",
      dimension: 3,
    });
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

describe("frontmatter metadata round-trip", () => {
  it("fmType/fmTitle/fmTags/fmSources survive upsertFile + similaritySearch", async () => {
    const dir = await makeTmpDir();
    const store = openSqliteStore(testConfig(dir));
    store.initializeSchema({
      embeddingProvider: "openai",
      embeddingModel: "m",
      dimension: 3,
    });

    const scanned: ScannedFile = {
      absolutePath: "/repos/docs/guide.md",
      relativePath: "docs/guide.md",
      repo: "docs",
      language: "md",
      content: [
        "---",
        "type: doc",
        "title: Guide Title",
        "tags:",
        "  - a",
        "  - b",
        "sources:",
        "  - src1",
        "---",
        "",
        "Body content.",
      ].join("\n"),
      contentHash: "h1",
    };

    const docs = await splitFile(scanned);
    expect(docs.length).toBeGreaterThanOrEqual(1);
    const embedding = normalized([1, 0, 0]);
    const entries: StoredEntry[] = docs.map((doc) => ({
      embedding,
      pageContent: doc.pageContent,
      metadata: doc.metadata,
    }));

    store.upsertFile("docs", "docs/guide.md", "h1", entries);

    const results = store.similaritySearch(embedding, entries.length);
    expect(results).toHaveLength(entries.length);
    for (const result of results) {
      expect(result.metadata.fmType).toBe("doc");
      expect(result.metadata.fmTitle).toBe("Guide Title");
      expect(result.metadata.fmTags).toEqual(["a", "b"]);
      expect(result.metadata.fmSources).toEqual(["src1"]);
    }

    const fileMetadata = store.getFileMetadata("docs", "docs/guide.md");
    expect(fileMetadata?.fmType).toBe("doc");
    expect(fileMetadata?.fmTitle).toBe("Guide Title");
    expect(fileMetadata?.fmTags).toEqual(["a", "b"]);
    expect(fileMetadata?.fmSources).toEqual(["src1"]);

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
      s1.initializeSchema({
        embeddingProvider: "openai",
        embeddingModel: "m",
        dimension: 4,
      });
      s2.initializeSchema({
        embeddingProvider: "openai",
        embeddingModel: "m",
        dimension: 4,
      });
      expect(s1.getMeta()?.dimension).toBe(4);
      expect(s2.getMeta()?.dimension).toBe(4);
      // A racing call with a DIFFERENT dim throws once the store is populated.
      expect(() =>
        s2.initializeSchema({
          embeddingProvider: "openai",
          embeddingModel: "m",
          dimension: 8,
        }),
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
    writer.initializeSchema({
      embeddingProvider: "openai",
      embeddingModel: "m",
      dimension: 3,
    });
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
    writer.initializeSchema({
      embeddingProvider: "openai",
      embeddingModel: "m",
      dimension: 3,
    });
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

    const result = spawnSync(process.execPath, [scriptPath], {
      encoding: "utf8",
    });
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
