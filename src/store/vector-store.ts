import type { Embeddings } from "@langchain/core/embeddings";
import { Document } from "@langchain/core/documents";
import type { Config } from "../config.js";
import { mkdir, writeFile, readFile, rename, rm } from "node:fs/promises";
import { join } from "node:path";
import { existsSync, createReadStream, createWriteStream } from "node:fs";
import { createInterface } from "node:readline";
import { once } from "node:events";

const EMBEDDINGS_FILE = "embeddings.jsonl";
const LEGACY_EMBEDDINGS_FILE = "embeddings.json";

export interface StoredVector {
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

function getEmbeddingsPath(config: Config): string {
  return join(config.dataDir, EMBEDDINGS_FILE);
}

function getLegacyEmbeddingsPath(config: Config): string {
  return join(config.dataDir, LEGACY_EMBEDDINGS_FILE);
}

async function loadStoredVectorsFromJsonl(path: string): Promise<StoredVector[]> {
  const vectors: StoredVector[] = [];
  const rl = createInterface({
    input: createReadStream(path, { encoding: "utf8" }),
    crlfDelay: Infinity,
  });

  for await (const line of rl) {
    const trimmed = line.trim();
    if (!trimmed) continue;

    try {
      const parsed = JSON.parse(trimmed) as {
        type?: string;
        embedding?: number[];
        doc?: { pageContent: string; metadata: Record<string, unknown> };
      };
      if (parsed.type === "meta") continue;
      if (!Array.isArray(parsed.embedding) || !parsed.doc) continue;
      vectors.push(parsed as StoredVector);
    } catch {
      // Ignore malformed lines and continue loading valid vectors.
    }
  }

  return vectors;
}

async function loadStoredVectorsFromLegacyJson(path: string): Promise<StoredVector[]> {
  const raw = await readFile(path, "utf-8");
  const data = JSON.parse(raw) as { vectors?: StoredVector[] };
  return Array.isArray(data.vectors) ? data.vectors : [];
}

async function writeJsonlVectors(
  path: string,
  vectors: StoredVector[],
  options: { append: boolean; includeMeta: boolean; totalCount?: number },
): Promise<void> {
  const stream = createWriteStream(path, {
    encoding: "utf8",
    flags: options.append ? "a" : "w",
  });

  const writeLine = async (line: string) => {
    if (stream.write(`${line}\n`)) return;
    await once(stream, "drain");
  };

  try {
    if (options.includeMeta) {
      await writeLine(
        JSON.stringify({
          type: "meta",
          embeddedAt: new Date().toISOString(),
          count: options.totalCount ?? vectors.length,
        }),
      );
    }
    for (const vector of vectors) {
      await writeLine(JSON.stringify(vector));
    }
  } finally {
    await new Promise<void>((resolve, reject) => {
      stream.on("finish", resolve);
      stream.on("error", reject);
      stream.end();
    });
  }
}

export async function loadStoredVectors(config: Config): Promise<StoredVector[]> {
  if (config.vectorStoreType !== "directory") return [];

  const jsonlPath = getEmbeddingsPath(config);
  const legacyPath = getLegacyEmbeddingsPath(config);
  if (!existsSync(jsonlPath) && !existsSync(legacyPath)) return [];

  try {
    if (existsSync(jsonlPath)) {
      return await loadStoredVectorsFromJsonl(jsonlPath);
    }
    return await loadStoredVectorsFromLegacyJson(legacyPath);
  } catch (err) {
    console.warn("Failed to load cached embeddings:", err);
    return [];
  }
}

export async function persistStoredVectors(
  vectors: StoredVector[],
  config: Config,
): Promise<void> {
  if (config.vectorStoreType !== "directory") return;

  await mkdir(config.dataDir, { recursive: true });
  const embPath = getEmbeddingsPath(config);
  const legacyPath = getLegacyEmbeddingsPath(config);
  const tempPath = `${embPath}.tmp`;

  await writeJsonlVectors(tempPath, vectors, {
    append: false,
    includeMeta: true,
    totalCount: vectors.length,
  });
  await rename(tempPath, embPath);
  if (existsSync(legacyPath)) {
    await rm(legacyPath, { force: true });
  }
  console.log(`Persisted ${vectors.length} vectors to ${embPath}`);
}

export async function initializeStoredVectors(
  vectors: StoredVector[],
  config: Config,
): Promise<void> {
  if (config.vectorStoreType !== "directory") return;

  await mkdir(config.dataDir, { recursive: true });
  const embPath = getEmbeddingsPath(config);
  const legacyPath = getLegacyEmbeddingsPath(config);
  await writeJsonlVectors(embPath, vectors, {
    append: false,
    includeMeta: true,
    totalCount: vectors.length,
  });
  if (existsSync(legacyPath)) {
    await rm(legacyPath, { force: true });
  }
}

export async function appendStoredVectors(
  vectors: StoredVector[],
  config: Config,
): Promise<void> {
  if (config.vectorStoreType !== "directory") return;
  if (vectors.length === 0) return;

  await mkdir(config.dataDir, { recursive: true });
  const embPath = getEmbeddingsPath(config);
  await writeJsonlVectors(embPath, vectors, {
    append: true,
    includeMeta: false,
  });
}

export async function createVectorStore(
  embeddings: Embeddings,
  config: Config,
): Promise<VectorStoreWrapper> {
  const vectors: StoredVector[] = await loadStoredVectors(config);

  if (vectors.length > 0) {
    console.log(`Loaded ${vectors.length} embedded vectors from cache`);
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
      await persistStoredVectors(vectors, config);
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
