import { afterEach, describe, it, expect, vi } from "vitest";
import { Document } from "@langchain/core/documents";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  aggregateIndexedRepos,
  assertCompatibleIndex,
  createVectorStore,
  IndexFingerprintError,
  initializeStoredVectors,
  loadStoredVectorsWithMeta,
  persistStoredVectors,
  type StoredVector,
} from "../../src/store/vector-store.js";
import type { Config } from "../../src/config.js";
import type { Embeddings } from "@langchain/core/embeddings";

// Deterministic fake embeddings for testing (no API calls)
function fakeEmbeddings(): Embeddings {
  let callCount = 0;

  function deterministicVector(text: string): number[] {
    // Simple hash-based vector for reproducible similarity
    const vec = new Array(8).fill(0);
    for (let i = 0; i < text.length; i++) {
      vec[i % 8] += text.charCodeAt(i) / 1000;
    }
    // Normalize
    const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
    return vec.map((v) => v / (norm || 1));
  }

  return {
    embedDocuments: async (texts: string[]) => texts.map(deterministicVector),
    embedQuery: async (text: string) => deterministicVector(text),
  } as unknown as Embeddings;
}

const testConfig: Config = {
  scanRoot: "/tmp/test",
  dataDir: "/tmp/oracle-test-data",
  embeddingProvider: "openai",
  llmProvider: "auto",
  ollamaBaseUrl: "http://localhost:11434/v1",
  embeddingModel: "test",
  llmModel: "test",
  vectorStoreType: "memory",
};

function directoryConfig(dataDir: string, overrides: Partial<Config> = {}): Config {
  return {
    ...testConfig,
    dataDir,
    vectorStoreType: "directory",
    ...overrides,
  };
}

function constantVector(length: number, seed = 0.1): number[] {
  return new Array(length).fill(seed);
}

describe("createVectorStore", () => {
  it("adds documents and retrieves by similarity", async () => {
    const store = await createVectorStore(fakeEmbeddings(), testConfig);

    await store.addDocuments([
      new Document({ pageContent: "function handleLogin() { ... }", metadata: { repo: "auth", filePath: "auth/login.ts" } }),
      new Document({ pageContent: "function handlePayment() { ... }", metadata: { repo: "billing", filePath: "billing/pay.ts" } }),
      new Document({ pageContent: "function authenticateUser() { ... }", metadata: { repo: "auth", filePath: "auth/middleware.ts" } }),
    ]);

    const results = await store.similaritySearch("authentication login", 2);
    expect(results).toHaveLength(2);
    // Results should be Document instances with metadata
    expect(results[0].metadata).toHaveProperty("repo");
    expect(results[0].metadata).toHaveProperty("filePath");
  });

  it("filters by metadata", async () => {
    const store = await createVectorStore(fakeEmbeddings(), testConfig);

    await store.addDocuments([
      new Document({ pageContent: "auth code", metadata: { repo: "auth", filePath: "auth/x.ts" } }),
      new Document({ pageContent: "billing code", metadata: { repo: "billing", filePath: "billing/x.ts" } }),
    ]);

    const filtered = await store.similaritySearch("code", 10, { repo: "billing" });
    expect(filtered).toHaveLength(1);
    expect(filtered[0].metadata.repo).toBe("billing");
  });

  it("returns empty array when no documents indexed", async () => {
    const store = await createVectorStore(fakeEmbeddings(), testConfig);
    const results = await store.similaritySearch("anything");
    expect(results).toEqual([]);
  });

  it("respects k limit", async () => {
    const store = await createVectorStore(fakeEmbeddings(), testConfig);

    await store.addDocuments(
      Array.from({ length: 20 }, (_, i) =>
        new Document({ pageContent: `chunk ${i}`, metadata: { repo: "r", filePath: `r/${i}.ts` } }),
      ),
    );

    const results = await store.similaritySearch("chunk", 3);
    expect(results).toHaveLength(3);
  });

  it("listRepos reports chunk and file counts per repo", async () => {
    const store = await createVectorStore(fakeEmbeddings(), testConfig);

    await store.addDocuments([
      new Document({ pageContent: "a", metadata: { repo: "auth", filePath: "auth/login.ts" } }),
      new Document({ pageContent: "b", metadata: { repo: "auth", filePath: "auth/login.ts" } }),
      new Document({ pageContent: "c", metadata: { repo: "auth", filePath: "auth/middleware.ts" } }),
      new Document({ pageContent: "d", metadata: { repo: "billing", filePath: "billing/pay.ts" } }),
    ]);

    expect(store.listRepos()).toEqual([
      { repo: "auth", chunkCount: 3, fileCount: 2 },
      { repo: "billing", chunkCount: 1, fileCount: 1 },
    ]);
  });
});

describe("aggregateIndexedRepos", () => {
  function vec(repo: string | undefined, filePath?: string): StoredVector {
    const metadata: Record<string, unknown> = {};
    if (repo !== undefined) metadata.repo = repo;
    if (filePath !== undefined) metadata.filePath = filePath;
    return { embedding: [0], doc: { pageContent: "x", metadata } };
  }

  it("returns [] for empty input", () => {
    expect(aggregateIndexedRepos([])).toEqual([]);
  });

  it("skips vectors without a repo", () => {
    expect(aggregateIndexedRepos([vec(undefined, "x.ts")])).toEqual([]);
  });

  it("counts a chunk without filePath as a chunk, not a file", () => {
    expect(aggregateIndexedRepos([vec("r"), vec("r", "x.ts")])).toEqual([
      { repo: "r", chunkCount: 2, fileCount: 1 },
    ]);
  });
});

describe("embedding fingerprint", () => {
  const tmpDirs: string[] = [];

  async function makeTmpDir(): Promise<string> {
    const dir = await mkdtemp(join(tmpdir(), "oracle-fp-"));
    tmpDirs.push(dir);
    return dir;
  }

  function sampleVector(dimension: number): StoredVector {
    return {
      embedding: constantVector(dimension, 0.25),
      doc: {
        pageContent: "hello",
        metadata: { repo: "r", filePath: "r/x.ts" },
      },
    };
  }

  afterEach(async () => {
    while (tmpDirs.length > 0) {
      const dir = tmpDirs.pop();
      if (!dir) continue;
      await rm(dir, { recursive: true, force: true });
    }
  });

  it("writes provider, model and dimension into the meta line", async () => {
    const dir = await makeTmpDir();
    const config = directoryConfig(dir, {
      embeddingProvider: "openai",
      embeddingModel: "text-embedding-3-small",
    });

    await initializeStoredVectors([sampleVector(1536)], config);

    const { vectors, meta } = await loadStoredVectorsWithMeta(config);
    expect(vectors).toHaveLength(1);
    expect(meta).not.toBeNull();
    expect(meta?.embeddingProvider).toBe("openai");
    expect(meta?.embeddingModel).toBe("text-embedding-3-small");
    expect(meta?.dimension).toBe(1536);
    expect(meta?.count).toBe(1);
  });

  it("persistStoredVectors round-trips the fingerprint", async () => {
    const dir = await makeTmpDir();
    const config = directoryConfig(dir, {
      embeddingProvider: "ollama",
      embeddingModel: "nomic-embed-text",
    });

    await persistStoredVectors([sampleVector(768), sampleVector(768)], config);

    const { meta } = await loadStoredVectorsWithMeta(config);
    expect(meta?.embeddingProvider).toBe("ollama");
    expect(meta?.embeddingModel).toBe("nomic-embed-text");
    expect(meta?.dimension).toBe(768);
  });

  it("rejects a loaded index whose provider does not match config", async () => {
    const dir = await makeTmpDir();
    const writerConfig = directoryConfig(dir, {
      embeddingProvider: "openai",
      embeddingModel: "text-embedding-3-small",
    });
    await initializeStoredVectors([sampleVector(1536)], writerConfig);

    const readerConfig = directoryConfig(dir, {
      embeddingProvider: "ollama",
      embeddingModel: "nomic-embed-text",
    });
    const { vectors, meta } = await loadStoredVectorsWithMeta(readerConfig);
    expect(() => assertCompatibleIndex(vectors, meta, readerConfig)).toThrow(
      IndexFingerprintError,
    );
    expect(() => assertCompatibleIndex(vectors, meta, readerConfig)).toThrow(
      /provider "openai"/,
    );
  });

  it("rejects a loaded index whose model does not match config", async () => {
    const dir = await makeTmpDir();
    const writerConfig = directoryConfig(dir, {
      embeddingProvider: "openai",
      embeddingModel: "text-embedding-3-small",
    });
    await initializeStoredVectors([sampleVector(1536)], writerConfig);

    const readerConfig = directoryConfig(dir, {
      embeddingProvider: "openai",
      embeddingModel: "text-embedding-3-large",
    });
    const { vectors, meta } = await loadStoredVectorsWithMeta(readerConfig);
    expect(() => assertCompatibleIndex(vectors, meta, readerConfig)).toThrow(
      /text-embedding-3-small/,
    );
  });

  it("rejects an index whose meta dimension differs from stored vectors", async () => {
    const dir = await makeTmpDir();
    const config = directoryConfig(dir, {
      embeddingProvider: "openai",
      embeddingModel: "text-embedding-3-small",
    });
    const jsonlPath = join(dir, "embeddings.jsonl");
    const metaLine = JSON.stringify({
      type: "meta",
      embeddedAt: new Date().toISOString(),
      count: 1,
      embeddingProvider: "openai",
      embeddingModel: "text-embedding-3-small",
      dimension: 1536,
    });
    const vectorLine = JSON.stringify(sampleVector(768));
    await writeFile(jsonlPath, `${metaLine}\n${vectorLine}\n`, "utf8");

    const { vectors, meta } = await loadStoredVectorsWithMeta(config);
    expect(() => assertCompatibleIndex(vectors, meta, config)).toThrow(
      /meta says dimension 1536 but first vector has 768/,
    );
  });

  it("legacy index without meta logs a warning and keeps loading", async () => {
    const dir = await makeTmpDir();
    const config = directoryConfig(dir, {
      embeddingProvider: "openai",
      embeddingModel: "text-embedding-3-small",
    });
    const jsonlPath = join(dir, "embeddings.jsonl");
    await writeFile(jsonlPath, `${JSON.stringify(sampleVector(1536))}\n`, "utf8");

    const { vectors, meta } = await loadStoredVectorsWithMeta(config);
    expect(meta).toBeNull();
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      expect(() => assertCompatibleIndex(vectors, meta, config)).not.toThrow();
      expect(warnSpy).toHaveBeenCalledWith(
        expect.stringContaining("no embedding fingerprint"),
      );
    } finally {
      warnSpy.mockRestore();
    }
  });

  it("meta without dimension (partial legacy meta) is still accepted when provider+model match", async () => {
    const dir = await makeTmpDir();
    const config = directoryConfig(dir, {
      embeddingProvider: "openai",
      embeddingModel: "text-embedding-3-small",
    });
    const jsonlPath = join(dir, "embeddings.jsonl");
    const metaLine = JSON.stringify({
      type: "meta",
      embeddedAt: new Date().toISOString(),
      count: 1,
      embeddingProvider: "openai",
      embeddingModel: "text-embedding-3-small",
    });
    const vectorLine = JSON.stringify(sampleVector(1536));
    await writeFile(jsonlPath, `${metaLine}\n${vectorLine}\n`, "utf8");

    const { vectors, meta } = await loadStoredVectorsWithMeta(config);
    expect(meta?.dimension).toBeUndefined();
    expect(() => assertCompatibleIndex(vectors, meta, config)).not.toThrow();
  });

  it("createVectorStore throws at construction time when fingerprint mismatches", async () => {
    const dir = await makeTmpDir();
    const writerConfig = directoryConfig(dir, {
      embeddingProvider: "openai",
      embeddingModel: "text-embedding-3-small",
    });
    await initializeStoredVectors([sampleVector(1536)], writerConfig);

    const readerConfig = directoryConfig(dir, {
      embeddingProvider: "ollama",
      embeddingModel: "nomic-embed-text",
    });
    await expect(createVectorStore(fakeEmbeddings(), readerConfig)).rejects.toBeInstanceOf(
      IndexFingerprintError,
    );
  });

  it("fresh index rewrite (no reused vectors) captures dimension in final meta line", async () => {
    // Simulates the cold-start index flow in src/index.ts:
    //   initializeStoredVectors([], config) → no dim in initial meta
    //   appendStoredVectors([...], config)  → no meta rewrite
    //   persistStoredVectors([...], config) → final atomic rewrite with full meta
    const dir = await makeTmpDir();
    const config = directoryConfig(dir, {
      embeddingProvider: "openai",
      embeddingModel: "text-embedding-3-small",
    });

    await initializeStoredVectors([], config);
    const afterInit = await loadStoredVectorsWithMeta(config);
    expect(afterInit.meta?.dimension).toBeUndefined();

    // appendStoredVectors is imported transitively via the store module;
    // we call persistStoredVectors directly (what index.ts does on cold start).
    const freshVectors = [sampleVector(1536), sampleVector(1536)];
    await persistStoredVectors(freshVectors, config);

    const afterPersist = await loadStoredVectorsWithMeta(config);
    expect(afterPersist.meta?.embeddingProvider).toBe("openai");
    expect(afterPersist.meta?.embeddingModel).toBe("text-embedding-3-small");
    expect(afterPersist.meta?.dimension).toBe(1536);
    expect(afterPersist.vectors).toHaveLength(2);
  });

  it("addDocuments refuses to mix dimensions mid-session", async () => {
    // Build a store with legacy index (no meta) at dim 16.
    const dir = await makeTmpDir();
    const config = directoryConfig(dir);
    const jsonlPath = join(dir, "embeddings.jsonl");
    await writeFile(jsonlPath, `${JSON.stringify(sampleVector(16))}\n`, "utf8");

    // fakeEmbeddings produces dim-8 vectors — a mismatch with the dim-16
    // already stored. addDocuments must refuse before persisting.
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const logSpy = vi.spyOn(console, "log").mockImplementation(() => {});
    try {
      const store = await createVectorStore(fakeEmbeddings(), config);
      await expect(
        store.addDocuments([
          new Document({ pageContent: "x", metadata: { repo: "r", filePath: "r/x.ts" } }),
        ]),
      ).rejects.toBeInstanceOf(IndexFingerprintError);
    } finally {
      warnSpy.mockRestore();
      logSpy.mockRestore();
    }
  });

  it("similaritySearch throws when query embedding dim does not match stored vectors (legacy path)", async () => {
    const dir = await makeTmpDir();
    const config = directoryConfig(dir);
    const jsonlPath = join(dir, "embeddings.jsonl");
    // Legacy: no meta line; stored vectors have dimension 32.
    await writeFile(jsonlPath, `${JSON.stringify(sampleVector(32))}\n`, "utf8");

    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const logSpy = vi.spyOn(console, "log").mockImplementation(() => {});
    try {
      const store = await createVectorStore(fakeEmbeddings(), config);
      // fakeEmbeddings produces dimension-8 vectors, so the query dim (8)
      // will not match the stored dim (32).
      await expect(store.similaritySearch("anything")).rejects.toBeInstanceOf(
        IndexFingerprintError,
      );
    } finally {
      warnSpy.mockRestore();
      logSpy.mockRestore();
    }
  });
});
