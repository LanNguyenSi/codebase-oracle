import { existsSync } from "node:fs";
import { rename } from "node:fs/promises";
import { createReadStream } from "node:fs";
import { createInterface } from "node:readline";
import { join } from "node:path";
import type { Config } from "./config.js";
import { openSqliteStore, type StoredEntry } from "./store/sqlite-store.js";

const JSONL_FILE = "embeddings.jsonl";
const BACKUP_FILE = ".embeddings.jsonl.bak";

interface ParsedLine {
  meta?: {
    embeddingProvider?: string;
    embeddingModel?: string;
    dimension?: number;
  };
  vector?: StoredEntry;
}

function parseLine(line: string): ParsedLine | null {
  const trimmed = line.trim();
  if (!trimmed) return null;
  try {
    const parsed = JSON.parse(trimmed) as {
      type?: string;
      embedding?: number[];
      doc?: { pageContent: string; metadata: Record<string, unknown> };
      embeddingProvider?: unknown;
      embeddingModel?: unknown;
      dimension?: unknown;
    };
    if (parsed.type === "meta") {
      return {
        meta: {
          embeddingProvider:
            typeof parsed.embeddingProvider === "string" ? parsed.embeddingProvider : undefined,
          embeddingModel:
            typeof parsed.embeddingModel === "string" ? parsed.embeddingModel : undefined,
          dimension: typeof parsed.dimension === "number" ? parsed.dimension : undefined,
        },
      };
    }
    if (!Array.isArray(parsed.embedding) || !parsed.doc) return null;
    return {
      vector: {
        embedding: parsed.embedding,
        pageContent: parsed.doc.pageContent,
        metadata: parsed.doc.metadata,
      },
    };
  } catch {
    return null;
  }
}

async function scanJsonlForDimensions(path: string): Promise<{
  dimensions: Set<number>;
  vectorCount: number;
  metaProvider?: string;
  metaModel?: string;
  metaDimension?: number;
}> {
  const rl = createInterface({
    input: createReadStream(path, { encoding: "utf8" }),
    crlfDelay: Infinity,
  });
  const dimensions = new Set<number>();
  let vectorCount = 0;
  let metaProvider: string | undefined;
  let metaModel: string | undefined;
  let metaDimension: number | undefined;
  for await (const line of rl) {
    const parsed = parseLine(line);
    if (!parsed) continue;
    if (parsed.meta) {
      metaProvider = parsed.meta.embeddingProvider ?? metaProvider;
      metaModel = parsed.meta.embeddingModel ?? metaModel;
      metaDimension = parsed.meta.dimension ?? metaDimension;
      continue;
    }
    if (parsed.vector) {
      dimensions.add(parsed.vector.embedding.length);
      vectorCount++;
    }
  }
  return { dimensions, vectorCount, metaProvider, metaModel, metaDimension };
}

export async function runMigrateStore(config: Config): Promise<void> {
  const jsonlPath = join(config.dataDir, JSONL_FILE);
  const backupPath = join(config.dataDir, BACKUP_FILE);

  if (!existsSync(jsonlPath)) {
    console.log(`migrate-store: no ${jsonlPath} found. Nothing to migrate.`);
    return;
  }

  // Pre-scan so we refuse clearly before touching the store when the source
  // is ambiguous or corrupt (mixed dims, empty, meta-dim vs. vector-dim mismatch).
  const scan = await scanJsonlForDimensions(jsonlPath);
  if (scan.vectorCount === 0) {
    throw new Error(
      `migrate-store: ${jsonlPath} has no vector rows. Delete it manually if you want a clean store.`,
    );
  }
  if (scan.dimensions.size > 1) {
    const dims = Array.from(scan.dimensions).sort((a, b) => a - b).join(", ");
    throw new Error(
      `migrate-store: ${jsonlPath} has vectors of mixed dimensions (${dims}). Legacy index is corrupt; delete it and re-index.`,
    );
  }
  const fileDim = [...scan.dimensions][0];
  if (typeof scan.metaDimension === "number" && scan.metaDimension !== fileDim) {
    throw new Error(
      `migrate-store: ${jsonlPath} meta says dimension ${scan.metaDimension} but vectors are dimension ${fileDim}. Refusing to migrate a corrupt index.`,
    );
  }

  const store = openSqliteStore(config);
  try {
    if (store.count() > 0) {
      throw new Error(
        `migrate-store: refusing to overwrite existing store at ${store.dbPath} (has ${store.count()} chunks). Delete or move it first.`,
      );
    }

    let provider = scan.metaProvider;
    let model = scan.metaModel;
    if (!provider || !model) {
      console.warn(
        "migrate-store: JSONL has no embedding fingerprint (legacy format). Using current config (" +
          `${config.embeddingProvider} / ${config.embeddingModel}) — make sure this matches the data.`,
      );
      provider = config.embeddingProvider;
      model = config.embeddingModel;
    }
    store.initializeSchema({
      embeddingProvider: provider,
      embeddingModel: model,
      dimension: fileDim,
    });

    console.log(`migrate-store: reading ${jsonlPath}...`);
    const rl = createInterface({
      input: createReadStream(jsonlPath, { encoding: "utf8" }),
      crlfDelay: Infinity,
    });

    const batch: StoredEntry[] = [];
    const BATCH_SIZE = 500;
    let total = 0;

    const flush = () => {
      if (batch.length === 0) return;
      store.insertBatch(batch);
      total += batch.length;
      batch.length = 0;
      process.stdout.write(`  migrated ${total} chunks\r`);
    };

    for await (const line of rl) {
      const parsed = parseLine(line);
      if (!parsed || !parsed.vector) continue;
      batch.push(parsed.vector);
      if (batch.length >= BATCH_SIZE) flush();
    }
    flush();
    process.stdout.write("\n");

    console.log(
      `migrate-store: wrote ${total} chunks into ${store.dbPath} (provider=${provider}, model=${model}, dim=${fileDim}).`,
    );

    await rename(jsonlPath, backupPath);
    console.log(`migrate-store: moved ${jsonlPath} → ${backupPath}`);
  } finally {
    store.close();
  }
}
