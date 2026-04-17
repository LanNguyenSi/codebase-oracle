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

export interface IndexedRepo {
  repo: string;
  chunkCount: number;
  fileCount: number;
}

export interface IndexMeta {
  embeddingProvider?: string;
  embeddingModel?: string;
  dimension?: number;
  embeddedAt?: string;
  count?: number;
}

export class IndexFingerprintError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "IndexFingerprintError";
  }
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
  listRepos(): IndexedRepo[];
}

export function aggregateIndexedRepos(vectors: StoredVector[]): IndexedRepo[] {
  const stats = new Map<string, { chunks: number; files: Set<string> }>();

  for (const vector of vectors) {
    const metadata = vector.doc.metadata as { repo?: unknown; filePath?: unknown };
    const repo = typeof metadata.repo === "string" ? metadata.repo : null;
    if (!repo) continue;

    const filePath = typeof metadata.filePath === "string" ? metadata.filePath : null;
    const entry = stats.get(repo) ?? { chunks: 0, files: new Set<string>() };
    entry.chunks++;
    if (filePath) entry.files.add(filePath);
    stats.set(repo, entry);
  }

  return Array.from(stats.entries())
    .map(([repo, s]) => ({ repo, chunkCount: s.chunks, fileCount: s.files.size }))
    .sort((a, b) => a.repo.localeCompare(b.repo));
}

function getEmbeddingsPath(config: Config): string {
  return join(config.dataDir, EMBEDDINGS_FILE);
}

function getLegacyEmbeddingsPath(config: Config): string {
  return join(config.dataDir, LEGACY_EMBEDDINGS_FILE);
}

async function loadStoredVectorsFromJsonl(
  path: string,
): Promise<{ vectors: StoredVector[]; meta: IndexMeta | null }> {
  const vectors: StoredVector[] = [];
  let meta: IndexMeta | null = null;
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
        embeddingProvider?: unknown;
        embeddingModel?: unknown;
        dimension?: unknown;
        embeddedAt?: unknown;
        count?: unknown;
      };
      if (parsed.type === "meta") {
        meta = {
          embeddingProvider:
            typeof parsed.embeddingProvider === "string" ? parsed.embeddingProvider : undefined,
          embeddingModel:
            typeof parsed.embeddingModel === "string" ? parsed.embeddingModel : undefined,
          dimension: typeof parsed.dimension === "number" ? parsed.dimension : undefined,
          embeddedAt: typeof parsed.embeddedAt === "string" ? parsed.embeddedAt : undefined,
          count: typeof parsed.count === "number" ? parsed.count : undefined,
        };
        continue;
      }
      if (!Array.isArray(parsed.embedding) || !parsed.doc) continue;
      vectors.push(parsed as StoredVector);
    } catch {
      // Ignore malformed lines and continue loading valid vectors.
    }
  }

  return { vectors, meta };
}

async function loadStoredVectorsFromLegacyJson(path: string): Promise<StoredVector[]> {
  const raw = await readFile(path, "utf-8");
  const data = JSON.parse(raw) as { vectors?: StoredVector[] };
  return Array.isArray(data.vectors) ? data.vectors : [];
}

interface WriteJsonlOptions {
  append: boolean;
  includeMeta: boolean;
  totalCount?: number;
  meta?: { embeddingProvider: string; embeddingModel: string };
}

async function writeJsonlVectors(
  path: string,
  vectors: StoredVector[],
  options: WriteJsonlOptions,
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
      const firstDim = vectors.find((v) => v.embedding.length > 0)?.embedding.length;
      const metaLine: Record<string, unknown> = {
        type: "meta",
        embeddedAt: new Date().toISOString(),
        count: options.totalCount ?? vectors.length,
      };
      if (options.meta) {
        metaLine.embeddingProvider = options.meta.embeddingProvider;
        metaLine.embeddingModel = options.meta.embeddingModel;
      }
      if (typeof firstDim === "number") {
        metaLine.dimension = firstDim;
      }
      await writeLine(JSON.stringify(metaLine));
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

export function assertCompatibleIndex(
  vectors: StoredVector[],
  meta: IndexMeta | null,
  config: Config,
): void {
  if (vectors.length === 0) return;

  if (meta === null || !meta.embeddingProvider || !meta.embeddingModel) {
    console.warn(
      `codebase-oracle: index at ${config.dataDir} has no embedding fingerprint (legacy). ` +
        `Assuming model "${config.embeddingModel}" (${config.embeddingProvider}). ` +
        "Re-run 'npm run index' to upgrade; stale results are possible until then.",
    );
    const storedDim = vectors[0].embedding.length;
    if (storedDim === 0) {
      throw new IndexFingerprintError(
        `Corrupt index at ${config.dataDir}: first vector has zero dimension. ` +
          "Delete the data dir and re-index.",
      );
    }
    return;
  }

  if (meta.embeddingProvider && meta.embeddingProvider !== config.embeddingProvider) {
    throw new IndexFingerprintError(
      `Index at ${config.dataDir} was built with provider "${meta.embeddingProvider}" but config is "${config.embeddingProvider}". ` +
        `Delete ${config.dataDir} and re-index, or set ORACLE_EMBEDDING_PROVIDER=${meta.embeddingProvider}.`,
    );
  }

  if (meta.embeddingModel && meta.embeddingModel !== config.embeddingModel) {
    throw new IndexFingerprintError(
      `Index at ${config.dataDir} was built with model "${meta.embeddingModel}" but config is "${config.embeddingModel}". ` +
        `Delete ${config.dataDir} and re-index, or set ORACLE_EMBEDDING_MODEL=${meta.embeddingModel}.`,
    );
  }

  const storedDim = vectors[0].embedding.length;
  if (typeof meta.dimension === "number" && meta.dimension !== storedDim) {
    throw new IndexFingerprintError(
      `Corrupt index at ${config.dataDir}: meta says dimension ${meta.dimension} but first vector has ${storedDim}. ` +
        "Delete the data dir and re-index.",
    );
  }
}

export async function loadStoredVectorsWithMeta(
  config: Config,
): Promise<{ vectors: StoredVector[]; meta: IndexMeta | null }> {
  if (config.vectorStoreType !== "directory") return { vectors: [], meta: null };

  const jsonlPath = getEmbeddingsPath(config);
  const legacyPath = getLegacyEmbeddingsPath(config);
  if (!existsSync(jsonlPath) && !existsSync(legacyPath)) return { vectors: [], meta: null };

  try {
    if (existsSync(jsonlPath)) {
      return await loadStoredVectorsFromJsonl(jsonlPath);
    }
    const vectors = await loadStoredVectorsFromLegacyJson(legacyPath);
    return { vectors, meta: null };
  } catch (err) {
    console.warn("Failed to load cached embeddings:", err);
    return { vectors: [], meta: null };
  }
}

export async function loadStoredVectors(config: Config): Promise<StoredVector[]> {
  const { vectors } = await loadStoredVectorsWithMeta(config);
  return vectors;
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
    meta: {
      embeddingProvider: config.embeddingProvider,
      embeddingModel: config.embeddingModel,
    },
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
    meta: {
      embeddingProvider: config.embeddingProvider,
      embeddingModel: config.embeddingModel,
    },
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
  const { vectors, meta } = await loadStoredVectorsWithMeta(config);
  assertCompatibleIndex(vectors, meta, config);

  if (vectors.length > 0) {
    console.log(`Loaded ${vectors.length} embedded vectors from cache`);
  }

  let expectedDim: number | null = vectors[0]?.embedding.length ?? null;

  return {
    async addDocuments(docs: Document[]) {
      // Batch embed in chunks of 100
      const batchSize = 100;
      for (let i = 0; i < docs.length; i += batchSize) {
        const batch = docs.slice(i, i + batchSize);
        const texts = batch.map((d) => d.pageContent);
        const embs = await embeddings.embedDocuments(texts);
        if (embs.length > 0) {
          if (expectedDim === null) {
            expectedDim = embs[0].length;
          } else if (embs[0].length !== expectedDim) {
            throw new IndexFingerprintError(
              `Tried to add vectors of dimension ${embs[0].length} to a store of dimension ${expectedDim}. ` +
                `Embedding provider/model likely changed mid-session. ` +
                `Delete ${config.dataDir} and re-index, or restore the previous embedding model.`,
            );
          }
        }
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

      if (expectedDim !== null && queryEmb.length !== expectedDim) {
        throw new IndexFingerprintError(
          `Query embedding has dimension ${queryEmb.length} but index vectors have ${expectedDim}. ` +
            `Provider "${config.embeddingProvider}" model "${config.embeddingModel}" does not match the indexed corpus. ` +
            `Delete ${config.dataDir} and re-index, or restore the previous embedding model.`,
        );
      }

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

    listRepos() {
      return aggregateIndexedRepos(vectors);
    },
  };
}

export async function listIndexedRepos(config: Config): Promise<IndexedRepo[]> {
  // Intentionally bypasses assertCompatibleIndex: listing repos is metadata-only
  // and safe to expose even if the stored fingerprint does not match the current
  // embedding config. Query/search paths do enforce the guard.
  return aggregateIndexedRepos(await loadStoredVectors(config));
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
