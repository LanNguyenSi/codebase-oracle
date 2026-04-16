import { describe, it, expect } from "vitest";
import { Document } from "@langchain/core/documents";
import { aggregateIndexedRepos, createVectorStore, type StoredVector } from "../../src/store/vector-store.js";
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
  embeddingModel: "test",
  llmModel: "test",
  vectorStoreType: "memory",
};

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
