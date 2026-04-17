import type { Embeddings } from "@langchain/core/embeddings";
import { Document } from "@langchain/core/documents";
import type { Config } from "../config.js";
import {
  openSqliteStore,
  IndexFingerprintError,
  type IndexedRepo,
  type SqliteStore,
  type StoredEntry,
} from "./sqlite-store.js";

export { IndexFingerprintError } from "./sqlite-store.js";
export type { IndexedRepo, StoreMeta as IndexMeta } from "./sqlite-store.js";

export interface VectorStoreWrapper {
  addDocuments(docs: Document[]): Promise<void>;
  similaritySearch(
    query: string,
    k?: number,
    filter?: Record<string, string>,
  ): Promise<Document[]>;
  listRepos(): IndexedRepo[];
  close(): void;
}

async function embedInBatches(
  embeddings: Embeddings,
  docs: Document[],
  batchSize = 100,
  onProgress?: (done: number, total: number) => void,
): Promise<StoredEntry[]> {
  const entries: StoredEntry[] = [];
  for (let i = 0; i < docs.length; i += batchSize) {
    const batch = docs.slice(i, i + batchSize);
    const texts = batch.map((d) => d.pageContent);
    const embs = await embeddings.embedDocuments(texts);
    for (let j = 0; j < batch.length; j++) {
      entries.push({
        embedding: embs[j],
        pageContent: batch[j].pageContent,
        metadata: batch[j].metadata as Record<string, unknown>,
      });
    }
    onProgress?.(Math.min(i + batchSize, docs.length), docs.length);
  }
  return entries;
}

export async function createVectorStore(
  embeddings: Embeddings,
  config: Config,
  injectedStore?: SqliteStore,
): Promise<VectorStoreWrapper> {
  const store = injectedStore ?? openSqliteStore(config);
  store.assertCompatibleWithConfig(config);

  const meta = store.getMeta();
  const count = store.count();
  if (count > 0) {
    console.log(`Loaded ${count} embedded vectors from ${store.dbPath}`);
  }

  let expectedDim: number | null = meta?.dimension ?? null;

  return {
    async addDocuments(docs: Document[]) {
      if (docs.length === 0) return;
      const sample = await embeddings.embedDocuments([docs[0].pageContent]);
      if (sample.length === 0) return;
      const dim = sample[0].length;

      if (expectedDim === null) {
        store.initializeSchema({
          embeddingProvider: config.embeddingProvider,
          embeddingModel: config.embeddingModel,
          dimension: dim,
        });
        expectedDim = dim;
      } else if (dim !== expectedDim) {
        throw new IndexFingerprintError(
          `Tried to add vectors of dimension ${dim} to a store of dimension ${expectedDim}. ` +
            `Embedding provider/model likely changed mid-session. ` +
            `Delete ${store.dbPath} and re-index, or restore the previous embedding model.`,
        );
      }

      const firstEntry: StoredEntry = {
        embedding: sample[0],
        pageContent: docs[0].pageContent,
        metadata: docs[0].metadata as Record<string, unknown>,
      };
      const rest = await embedInBatches(embeddings, docs.slice(1), 100, (done, total) => {
        if (total > 100) process.stdout.write(`  Embedded ${done}/${total}\r`);
      });
      store.insertBatch([firstEntry, ...rest]);
      if (docs.length > 100) console.log();
    },

    async similaritySearch(query: string, k = 10, filter?: Record<string, string>) {
      if (store.count() === 0) return [];
      const queryEmb = await embeddings.embedQuery(query);
      if (expectedDim === null) {
        const m = store.getMeta();
        expectedDim = m?.dimension ?? queryEmb.length;
      }
      if (queryEmb.length !== expectedDim) {
        throw new IndexFingerprintError(
          `Query embedding has dimension ${queryEmb.length} but index vectors have ${expectedDim}. ` +
            `Provider "${config.embeddingProvider}" model "${config.embeddingModel}" does not match the indexed corpus. ` +
            `Delete ${store.dbPath} and re-index, or restore the previous embedding model.`,
        );
      }

      const storeFilter = filter && typeof filter.repo === "string" ? { repo: filter.repo } : undefined;
      const results = store.similaritySearch(queryEmb, k, storeFilter);

      // If the caller passed non-repo filters, post-filter in JS.
      const extraKeys = filter
        ? Object.keys(filter).filter((key) => key !== "repo")
        : [];
      const filtered = extraKeys.length
        ? results.filter((r) =>
            extraKeys.every((key) => r.metadata[key] === filter![key]),
          )
        : results;

      return filtered.map(
        (r) => new Document({ pageContent: r.pageContent, metadata: r.metadata }),
      );
    },

    listRepos() {
      return store.listRepos();
    },

    close() {
      if (!injectedStore) store.close();
    },
  };
}

export function listIndexedRepos(config: Config): IndexedRepo[] {
  const store = openSqliteStore(config);
  try {
    // Intentionally bypasses assertCompatibleWithConfig: listing repos is
    // metadata-only and safe to expose even when the stored fingerprint does
    // not match the current embedding config.
    return store.listRepos();
  } finally {
    store.close();
  }
}
