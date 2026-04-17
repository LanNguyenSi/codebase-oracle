import { afterEach, describe, it, expect, vi } from "vitest";
import { Document } from "@langchain/core/documents";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  createVectorStore,
  IndexFingerprintError,
  listIndexedRepos,
} from "../../src/store/vector-store.js";
import { openSqliteStore } from "../../src/store/sqlite-store.js";
import type { Config } from "../../src/config.js";
import type { Embeddings } from "@langchain/core/embeddings";

// Deterministic fake embeddings for testing (no API calls)
function fakeEmbeddings(dimension = 8): Embeddings {
  function deterministicVector(text: string): number[] {
    const vec = new Array(dimension).fill(0);
    for (let i = 0; i < text.length; i++) {
      vec[i % dimension] += text.charCodeAt(i) / 1000;
    }
    const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
    return vec.map((v) => v / (norm || 1));
  }

  return {
    embedDocuments: async (texts: string[]) => texts.map(deterministicVector),
    embedQuery: async (text: string) => deterministicVector(text),
  } as unknown as Embeddings;
}

const tmpDirs: string[] = [];
async function makeTmpDir(): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), "oracle-store-"));
  tmpDirs.push(dir);
  return dir;
}

function directoryConfig(dataDir: string, overrides: Partial<Config> = {}): Config {
  return {
    scanRoot: "/tmp/test",
    dataDir,
    embeddingProvider: "openai",
    llmProvider: "auto",
    ollamaBaseUrl: "http://localhost:11434/v1",
    embeddingModel: "test",
    llmModel: "test",
    vectorStoreType: "directory",
    ...overrides,
  };
}

afterEach(async () => {
  while (tmpDirs.length > 0) {
    const dir = tmpDirs.pop();
    if (!dir) continue;
    await rm(dir, { recursive: true, force: true });
  }
});

describe("createVectorStore", () => {
  it("adds documents and retrieves by similarity", async () => {
    const dir = await makeTmpDir();
    const config = directoryConfig(dir);
    const store = await createVectorStore(fakeEmbeddings(), config);

    await store.addDocuments([
      new Document({
        pageContent: "function handleLogin() { ... }",
        metadata: { repo: "auth", filePath: "auth/login.ts" },
      }),
      new Document({
        pageContent: "function handlePayment() { ... }",
        metadata: { repo: "billing", filePath: "billing/pay.ts" },
      }),
      new Document({
        pageContent: "function authenticateUser() { ... }",
        metadata: { repo: "auth", filePath: "auth/middleware.ts" },
      }),
    ]);

    const results = await store.similaritySearch("authentication login", 2);
    expect(results).toHaveLength(2);
    expect(results[0].metadata).toHaveProperty("repo");
    expect(results[0].metadata).toHaveProperty("filePath");
    store.close();
  });

  it("filters by repo metadata", async () => {
    const dir = await makeTmpDir();
    const config = directoryConfig(dir);
    const store = await createVectorStore(fakeEmbeddings(), config);

    await store.addDocuments([
      new Document({ pageContent: "auth code", metadata: { repo: "auth", filePath: "auth/x.ts" } }),
      new Document({
        pageContent: "billing code",
        metadata: { repo: "billing", filePath: "billing/x.ts" },
      }),
    ]);

    const filtered = await store.similaritySearch("code", 10, { repo: "billing" });
    expect(filtered).toHaveLength(1);
    expect(filtered[0].metadata.repo).toBe("billing");
    store.close();
  });

  it("returns empty array when no documents indexed", async () => {
    const dir = await makeTmpDir();
    const config = directoryConfig(dir);
    const store = await createVectorStore(fakeEmbeddings(), config);
    const results = await store.similaritySearch("anything");
    expect(results).toEqual([]);
    store.close();
  });

  it("respects k limit", async () => {
    const dir = await makeTmpDir();
    const config = directoryConfig(dir);
    const store = await createVectorStore(fakeEmbeddings(), config);

    await store.addDocuments(
      Array.from(
        { length: 20 },
        (_, i) =>
          new Document({
            pageContent: `chunk ${i}`,
            metadata: { repo: "r", filePath: `r/${i}.ts` },
          }),
      ),
    );

    const results = await store.similaritySearch("chunk", 3);
    expect(results).toHaveLength(3);
    store.close();
  });

  it("listRepos reports chunk and file counts per repo", async () => {
    const dir = await makeTmpDir();
    const config = directoryConfig(dir);
    const store = await createVectorStore(fakeEmbeddings(), config);

    await store.addDocuments([
      new Document({ pageContent: "a", metadata: { repo: "auth", filePath: "auth/login.ts" } }),
      new Document({ pageContent: "b", metadata: { repo: "auth", filePath: "auth/login.ts" } }),
      new Document({
        pageContent: "c",
        metadata: { repo: "auth", filePath: "auth/middleware.ts" },
      }),
      new Document({ pageContent: "d", metadata: { repo: "billing", filePath: "billing/pay.ts" } }),
    ]);

    expect(store.listRepos()).toEqual([
      { repo: "auth", chunkCount: 3, fileCount: 2 },
      { repo: "billing", chunkCount: 1, fileCount: 1 },
    ]);
    store.close();
  });
});

describe("listIndexedRepos", () => {
  it("returns [] for a fresh store", async () => {
    const dir = await makeTmpDir();
    const config = directoryConfig(dir);
    // Opening and closing the store creates the file.
    openSqliteStore(config).close();
    expect(listIndexedRepos(config)).toEqual([]);
  });

  it("ignores fingerprint and reads repo list directly", async () => {
    // Build with openai, read with ollama — listIndexedRepos should still work.
    const dir = await makeTmpDir();
    const writerConfig = directoryConfig(dir, {
      embeddingProvider: "openai",
      embeddingModel: "text-embedding-3-small",
    });
    const store = await createVectorStore(fakeEmbeddings(), writerConfig);
    await store.addDocuments([
      new Document({ pageContent: "x", metadata: { repo: "r", filePath: "r/x.ts" } }),
    ]);
    store.close();

    const readerConfig = directoryConfig(dir, {
      embeddingProvider: "ollama",
      embeddingModel: "nomic-embed-text",
    });
    expect(listIndexedRepos(readerConfig)).toEqual([
      { repo: "r", chunkCount: 1, fileCount: 1 },
    ]);
  });
});

describe("embedding fingerprint", () => {
  it("writes provider, model and dimension on first insert", async () => {
    const dir = await makeTmpDir();
    const config = directoryConfig(dir, {
      embeddingProvider: "openai",
      embeddingModel: "text-embedding-3-small",
    });
    const store = await createVectorStore(fakeEmbeddings(16), config);
    await store.addDocuments([
      new Document({ pageContent: "x", metadata: { repo: "r", filePath: "r/x.ts" } }),
    ]);
    store.close();

    const raw = openSqliteStore(config);
    const meta = raw.getMeta();
    raw.close();
    expect(meta).not.toBeNull();
    expect(meta?.embeddingProvider).toBe("openai");
    expect(meta?.embeddingModel).toBe("text-embedding-3-small");
    expect(meta?.dimension).toBe(16);
  });

  it("rejects a loaded store whose provider does not match config", async () => {
    const dir = await makeTmpDir();
    const writerConfig = directoryConfig(dir, {
      embeddingProvider: "openai",
      embeddingModel: "text-embedding-3-small",
    });
    const store = await createVectorStore(fakeEmbeddings(), writerConfig);
    await store.addDocuments([
      new Document({ pageContent: "x", metadata: { repo: "r", filePath: "r/x.ts" } }),
    ]);
    store.close();

    const readerConfig = directoryConfig(dir, {
      embeddingProvider: "ollama",
      embeddingModel: "nomic-embed-text",
    });
    await expect(createVectorStore(fakeEmbeddings(), readerConfig)).rejects.toBeInstanceOf(
      IndexFingerprintError,
    );
  });

  it("rejects a loaded store whose model does not match config", async () => {
    const dir = await makeTmpDir();
    const writerConfig = directoryConfig(dir, {
      embeddingProvider: "openai",
      embeddingModel: "text-embedding-3-small",
    });
    const store = await createVectorStore(fakeEmbeddings(), writerConfig);
    await store.addDocuments([
      new Document({ pageContent: "x", metadata: { repo: "r", filePath: "r/x.ts" } }),
    ]);
    store.close();

    const readerConfig = directoryConfig(dir, {
      embeddingProvider: "openai",
      embeddingModel: "text-embedding-3-large",
    });
    await expect(createVectorStore(fakeEmbeddings(), readerConfig)).rejects.toThrow(
      /text-embedding-3-small/,
    );
  });

  it("addDocuments refuses to mix dimensions mid-session", async () => {
    const dir = await makeTmpDir();
    const config = directoryConfig(dir);
    const storeA = await createVectorStore(fakeEmbeddings(8), config);
    await storeA.addDocuments([
      new Document({ pageContent: "x", metadata: { repo: "r", filePath: "r/x.ts" } }),
    ]);
    storeA.close();

    const logSpy = vi.spyOn(console, "log").mockImplementation(() => {});
    try {
      const storeB = await createVectorStore(fakeEmbeddings(16), config);
      await expect(
        storeB.addDocuments([
          new Document({ pageContent: "y", metadata: { repo: "r", filePath: "r/y.ts" } }),
        ]),
      ).rejects.toBeInstanceOf(IndexFingerprintError);
      storeB.close();
    } finally {
      logSpy.mockRestore();
    }
  });

  it("similaritySearch throws when query embedding dim does not match stored vectors", async () => {
    const dir = await makeTmpDir();
    const config = directoryConfig(dir);
    const storeA = await createVectorStore(fakeEmbeddings(32), config);
    await storeA.addDocuments([
      new Document({ pageContent: "x", metadata: { repo: "r", filePath: "r/x.ts" } }),
    ]);
    storeA.close();

    const logSpy = vi.spyOn(console, "log").mockImplementation(() => {});
    try {
      const storeB = await createVectorStore(fakeEmbeddings(8), config);
      await expect(storeB.similaritySearch("anything")).rejects.toBeInstanceOf(
        IndexFingerprintError,
      );
      storeB.close();
    } finally {
      logSpy.mockRestore();
    }
  });
});
