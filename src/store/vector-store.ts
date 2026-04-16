import type { Embeddings } from "@langchain/core/embeddings";
import { Document } from "@langchain/core/documents";
import type { Config } from "../config.js";
import { mkdir, writeFile, readFile } from "node:fs/promises";
import { join } from "node:path";
import { existsSync } from "node:fs";

interface StoredVector {
  embedding: number[];
  doc: { pageContent: string; metadata: Record<string, unknown> };
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

export interface VectorStoreWrapper {
  addDocuments(docs: Document[]): Promise<void>;
  similaritySearch(query: string, k?: number, filter?: Record<string, string>): Promise<Document[]>;
}

export async function createVectorStore(
  embeddings: Embeddings,
  config: Config,
): Promise<VectorStoreWrapper> {
  const vectors: StoredVector[] = [];

  // Try loading persisted embeddings
  if (config.vectorStoreType === "directory") {
    const embPath = join(config.dataDir, "embeddings.json");
    if (existsSync(embPath)) {
      try {
        const raw = await readFile(embPath, "utf-8");
        const data = JSON.parse(raw) as { vectors: StoredVector[] };
        vectors.push(...data.vectors);
        console.log(`Loaded ${vectors.length} embedded vectors from cache`);
      } catch (err) {
        console.warn("Failed to load cached embeddings:", err);
      }
    }
  }

  return {
    async addDocuments(docs: Document[]) {
      // Batch embed in chunks of 100
      const batchSize = 100;
      for (let i = 0; i < docs.length; i += batchSize) {
        const batch = docs.slice(i, i + batchSize);
        const texts = batch.map((d) => d.pageContent);
        const embs = await embeddings.embedDocuments(texts);
        for (let j = 0; j < batch.length; j++) {
          vectors.push({
            embedding: embs[j],
            doc: { pageContent: batch[j].pageContent, metadata: batch[j].metadata as Record<string, unknown> },
          });
        }
        if (docs.length > batchSize) {
          process.stdout.write(`  Embedded ${Math.min(i + batchSize, docs.length)}/${docs.length}\r`);
        }
      }
      if (docs.length > batchSize) console.log();

      // Persist embeddings
      if (config.vectorStoreType === "directory") {
        await mkdir(config.dataDir, { recursive: true });
        const embPath = join(config.dataDir, "embeddings.json");
        await writeFile(embPath, JSON.stringify({ vectors, embeddedAt: new Date().toISOString() }));
        console.log(`Persisted ${vectors.length} vectors to ${embPath}`);
      }
    },

    async similaritySearch(query: string, k = 10, filter?: Record<string, string>) {
      if (vectors.length === 0) return [];

      const queryEmb = await embeddings.embedQuery(query);

      let candidates = vectors;
      if (filter) {
        candidates = vectors.filter((v) =>
          Object.entries(filter).every(([key, value]) => v.doc.metadata[key] === value),
        );
      }

      const scored = candidates
        .map((v) => ({
          score: cosineSimilarity(queryEmb, v.embedding),
          doc: v.doc,
        }))
        .sort((a, b) => b.score - a.score)
        .slice(0, k);

      return scored.map(
        (s) => new Document({ pageContent: s.doc.pageContent, metadata: s.doc.metadata }),
      );
    },
  };
}

export async function persistIndex(
  docs: Document[],
  config: Config,
): Promise<void> {
  if (config.vectorStoreType !== "directory") return;

  await mkdir(config.dataDir, { recursive: true });
  const indexPath = join(config.dataDir, "index.json");
  const data = {
    indexedAt: new Date().toISOString(),
    docCount: docs.length,
    docs: docs.map((d) => ({
      pageContent: d.pageContent,
      metadata: d.metadata,
    })),
  };
  await writeFile(indexPath, JSON.stringify(data));
  console.log(`Persisted ${docs.length} document chunks to ${indexPath}`);
}
