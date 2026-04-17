import Database from "better-sqlite3";
import * as sqliteVec from "sqlite-vec";
import { mkdirSync } from "node:fs";
import { join } from "node:path";
import type { Config } from "../config.js";

export const SCHEMA_VERSION = "1";
export const STORE_FILE = "store.db";

export interface StoreMeta {
  embeddingProvider: string;
  embeddingModel: string;
  dimension: number;
  embeddedAt: string | null;
  schemaVersion: string;
}

export interface IndexedRepo {
  repo: string;
  chunkCount: number;
  fileCount: number;
}

export interface StoredEntry {
  embedding: number[];
  pageContent: string;
  metadata: Record<string, unknown>;
}

export interface SearchResult {
  pageContent: string;
  metadata: Record<string, unknown>;
  distance: number;
}

export interface FileSignature {
  repo: string;
  filePath: string;
  fileHash: string | null;
}

export class IndexFingerprintError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "IndexFingerprintError";
  }
}

export interface SqliteStore {
  getMeta(): StoreMeta | null;
  /**
   * Initializes the vec0 virtual table with the given dimension and writes the
   * meta row. Idempotent iff the stored meta matches; otherwise throws.
   */
  initializeSchema(meta: {
    embeddingProvider: string;
    embeddingModel: string;
    dimension: number;
  }): void;
  assertCompatibleWithConfig(config: Config): void;
  count(): number;
  listRepos(): IndexedRepo[];
  similaritySearch(
    queryEmbedding: number[],
    k: number,
    filter?: { repo?: string },
  ): SearchResult[];
  upsertFile(
    repo: string,
    filePath: string,
    fileHash: string | null,
    entries: StoredEntry[],
  ): { added: number; removed: number };
  deleteByFile(repo: string, filePath: string): number;
  deleteByRepo(repo: string): number;
  fileSignatures(): Map<string, FileSignature>;
  insertBatch(entries: StoredEntry[]): void;
  close(): void;
  /**
   * The underlying absolute path to the SQLite file. Exposed for tooling
   * (migrate command, tests).
   */
  readonly dbPath: string;
  /** Last-write heartbeat used by readers to avoid stale JS-level caches. */
  bumpWriteEpoch(): void;
  getWriteEpoch(): number;
}

export function storePathFor(config: Config): string {
  return join(config.dataDir, STORE_FILE);
}

function fileKey(repo: string, filePath: string): string {
  return `${repo}::${filePath}`;
}

function toFloat32Buffer(embedding: number[]): Buffer {
  const arr = new Float32Array(embedding.length);
  for (let i = 0; i < embedding.length; i++) arr[i] = embedding[i];
  return Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength);
}

function extractString(obj: Record<string, unknown>, key: string): string | null {
  const value = obj[key];
  return typeof value === "string" ? value : null;
}

function ensureDirFor(path: string): void {
  const dir = path.replace(/\/[^/]+$/, "");
  if (!dir) return;
  mkdirSync(dir, { recursive: true });
}

interface CompiledStatements {
  getMeta: Database.Statement;
  upsertMeta: Database.Statement;
  countDocs: Database.Statement;
  listRepos: Database.Statement;
  deleteDocsByFile: Database.Statement<[string, string]>;
  deleteDocsByRepo: Database.Statement<[string]>;
  insertDoc: Database.Statement;
  selectFileSignatures: Database.Statement;
  selectEpoch: Database.Statement;
  upsertEpoch: Database.Statement;
}

/** Statements that depend on the vec0 virtual table existing. Compiled lazily. */
interface VecStatements {
  deleteVecByFile: Database.Statement<[string, string]>;
  deleteVecByRepo: Database.Statement<[string]>;
  insertVec: Database.Statement;
}

export function openSqliteStore(config: Config): SqliteStore {
  const dbPath = storePathFor(config);
  ensureDirFor(dbPath);
  const db = new Database(dbPath);
  db.pragma("journal_mode = WAL");
  db.pragma("synchronous = NORMAL");
  // Wait up to 5s on a busy database instead of immediately throwing. Watch-
  // mode writes and index-time upserts can briefly contend with readers or a
  // second writer (concurrent watch + index); SQLite's default behavior is
  // to fail fast with SQLITE_BUSY, which would abort the transaction.
  db.pragma("busy_timeout = 5000");
  db.pragma("foreign_keys = ON");
  sqliteVec.load(db);

  db.exec(`
    CREATE TABLE IF NOT EXISTS meta (
      key TEXT PRIMARY KEY,
      value TEXT
    );
    CREATE TABLE IF NOT EXISTS docs (
      rowid INTEGER PRIMARY KEY AUTOINCREMENT,
      repo TEXT NOT NULL,
      file_path TEXT NOT NULL,
      page_content TEXT NOT NULL,
      metadata TEXT NOT NULL,
      file_hash TEXT
    );
    CREATE INDEX IF NOT EXISTS docs_repo_file ON docs(repo, file_path);
    CREATE INDEX IF NOT EXISTS docs_file_hash ON docs(file_hash);
  `);

  // The vec0 virtual table is created lazily because its dimension is locked
  // at creation time. A fresh store has no embeddings yet — the table is
  // created on the first initializeSchema() call.
  const metaRow = (): Record<string, string> => {
    const rows = db.prepare("SELECT key, value FROM meta").all() as Array<{
      key: string;
      value: string;
    }>;
    const out: Record<string, string> = {};
    for (const row of rows) out[row.key] = row.value;
    return out;
  };

  let vecStmts: VecStatements | null = null;

  function compileVecStatements(): VecStatements {
    return {
      deleteVecByFile: db.prepare(
        "DELETE FROM vectors WHERE rowid IN (SELECT rowid FROM docs WHERE repo=? AND file_path=?)",
      ),
      deleteVecByRepo: db.prepare(
        "DELETE FROM vectors WHERE rowid IN (SELECT rowid FROM docs WHERE repo=?)",
      ),
      insertVec: db.prepare("INSERT INTO vectors(rowid, embedding) VALUES (?, ?)"),
    };
  }

  function ensureVecTable(dimension: number): void {
    if (!Number.isInteger(dimension) || dimension <= 0 || dimension > 8192) {
      throw new IndexFingerprintError(
        `Refusing to create vec0 table with dimension ${dimension}: must be a positive integer <= 8192.`,
      );
    }
    db.exec(
      `CREATE VIRTUAL TABLE IF NOT EXISTS vectors USING vec0(embedding float[${dimension}] distance_metric=cosine)`,
    );
    vecStmts = compileVecStatements();
  }

  function vec(): VecStatements {
    if (!vecStmts) {
      throw new IndexFingerprintError(
        `Store at ${dbPath} has no embedding dimension yet. Call initializeSchema() before writing vectors.`,
      );
    }
    return vecStmts;
  }

  const existing = metaRow();
  if (existing.dimension) {
    ensureVecTable(Number(existing.dimension));
  }

  const stmts: CompiledStatements = {
    getMeta: db.prepare("SELECT key, value FROM meta"),
    upsertMeta: db.prepare(
      "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
    ),
    countDocs: db.prepare("SELECT COUNT(*) AS c FROM docs"),
    listRepos: db.prepare(
      "SELECT repo, COUNT(*) AS chunkCount, COUNT(DISTINCT file_path) AS fileCount FROM docs GROUP BY repo ORDER BY repo",
    ),
    deleteDocsByFile: db.prepare("DELETE FROM docs WHERE repo=? AND file_path=? RETURNING rowid"),
    deleteDocsByRepo: db.prepare("DELETE FROM docs WHERE repo=? RETURNING rowid"),
    insertDoc: db.prepare(
      "INSERT INTO docs (repo, file_path, page_content, metadata, file_hash) VALUES (?, ?, ?, ?, ?)",
    ),
    selectFileSignatures: db.prepare(
      "SELECT repo, file_path AS filePath, file_hash AS fileHash FROM docs",
    ),
    selectEpoch: db.prepare("SELECT value FROM meta WHERE key = 'writeEpoch'"),
    upsertEpoch: db.prepare(
      "INSERT INTO meta(key, value) VALUES('writeEpoch', ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
    ),
  };

  function getMetaInternal(): StoreMeta | null {
    const row = metaRow();
    if (!row.embeddingProvider || !row.embeddingModel || !row.dimension) return null;
    return {
      embeddingProvider: row.embeddingProvider,
      embeddingModel: row.embeddingModel,
      dimension: Number(row.dimension),
      embeddedAt: row.embeddedAt ?? null,
      schemaVersion: row.schemaVersion ?? SCHEMA_VERSION,
    };
  }

  function initializeSchemaInternal(meta: {
    embeddingProvider: string;
    embeddingModel: string;
    dimension: number;
  }): void {
    // Validate dimension up front so a bad embedding response can't reach the
    // vec0 CREATE path.
    if (!Number.isInteger(meta.dimension) || meta.dimension <= 0 || meta.dimension > 8192) {
      throw new IndexFingerprintError(
        `Refusing to initialize store with dimension ${meta.dimension}: must be a positive integer <= 8192.`,
      );
    }

    // Single IMMEDIATE transaction: acquire the write lock, re-check meta
    // under the lock, then either no-op (matching meta) or create the vec
    // table + insert meta. Without IMMEDIATE, two racing watchers could both
    // pass the pre-check and then serialise in an unpredictable order — the
    // loser might have already run CREATE VIRTUAL TABLE IF NOT EXISTS before
    // discovering the dimension disagrees.
    const nowIso = new Date().toISOString();
    const tx = db.transaction(() => {
      const currentMeta = getMetaInternal();
      if (currentMeta) {
        if (currentMeta.dimension !== meta.dimension) {
          throw new IndexFingerprintError(
            `Store at ${dbPath} was initialized with dimension ${currentMeta.dimension}; cannot switch to ${meta.dimension} without a fresh store.`,
          );
        }
        if (currentMeta.embeddingProvider !== meta.embeddingProvider) {
          throw new IndexFingerprintError(
            `Store at ${dbPath} was initialized with provider "${currentMeta.embeddingProvider}"; cannot switch to "${meta.embeddingProvider}" without a fresh store.`,
          );
        }
        if (currentMeta.embeddingModel !== meta.embeddingModel) {
          throw new IndexFingerprintError(
            `Store at ${dbPath} was initialized with model "${currentMeta.embeddingModel}"; cannot switch to "${meta.embeddingModel}" without a fresh store.`,
          );
        }
        // Matching meta: ensure our handle has compiled vec statements even
        // if this process opened the store after another process created it.
        if (!vecStmts) ensureVecTable(meta.dimension);
        return;
      }
      ensureVecTable(meta.dimension);
      stmts.upsertMeta.run("embeddingProvider", meta.embeddingProvider);
      stmts.upsertMeta.run("embeddingModel", meta.embeddingModel);
      stmts.upsertMeta.run("dimension", String(meta.dimension));
      stmts.upsertMeta.run("embeddedAt", nowIso);
      stmts.upsertMeta.run("schemaVersion", SCHEMA_VERSION);
    });
    tx.immediate();
  }

  function assertCompatibleWithConfigInternal(config: Config): void {
    const meta = getMetaInternal();
    const docCount = (stmts.countDocs.get() as { c: number }).c;
    if (!meta) {
      // Empty fresh store is fine. If docs exist without meta, it's corrupt —
      // but that cannot happen because initializeSchema is always called
      // before any insert. Treat it as empty.
      if (docCount > 0) {
        throw new IndexFingerprintError(
          `Store at ${dbPath} has ${docCount} docs but no fingerprint metadata. Delete the store and re-index.`,
        );
      }
      return;
    }

    if (meta.embeddingProvider !== config.embeddingProvider) {
      throw new IndexFingerprintError(
        `Store at ${dbPath} was built with provider "${meta.embeddingProvider}" but config is "${config.embeddingProvider}". ` +
          `Delete ${dbPath} and re-index, or set ORACLE_EMBEDDING_PROVIDER=${meta.embeddingProvider}.`,
      );
    }

    if (meta.embeddingModel !== config.embeddingModel) {
      throw new IndexFingerprintError(
        `Store at ${dbPath} was built with model "${meta.embeddingModel}" but config is "${config.embeddingModel}". ` +
          `Delete ${dbPath} and re-index, or set ORACLE_EMBEDDING_MODEL=${meta.embeddingModel}.`,
      );
    }
  }

  function listReposInternal(): IndexedRepo[] {
    return stmts.listRepos.all() as IndexedRepo[];
  }

  function similaritySearchInternal(
    queryEmbedding: number[],
    k: number,
    filter?: { repo?: string },
  ): SearchResult[] {
    const meta = getMetaInternal();
    if (!meta) return [];
    if (queryEmbedding.length !== meta.dimension) {
      throw new IndexFingerprintError(
        `Query embedding has dimension ${queryEmbedding.length} but store dimension is ${meta.dimension}. ` +
          "Embedding provider/model has changed; delete the store and re-index.",
      );
    }

    const buf = toFloat32Buffer(queryEmbedding);
    // When filtering, the KNN search happens against all vectors, then we
    // intersect with the filter in the join. Overshoot the candidate pool so
    // filtered results don't go empty just because the nearest neighbors all
    // live in a different repo. 10x multiplier is generous for small filters
    // and still bounded — the total corpus is the upper bound.
    const candidateK = filter?.repo ? Math.max(k * 10, 100) : k;

    if (filter?.repo) {
      const sql = `
        SELECT d.repo AS repo,
               d.file_path AS file_path,
               d.page_content AS page_content,
               d.metadata AS metadata,
               m.distance AS distance
        FROM (SELECT rowid, distance FROM vectors WHERE embedding MATCH ? AND k = ?) m
        JOIN docs d ON d.rowid = m.rowid
        WHERE d.repo = ?
        ORDER BY m.distance
        LIMIT ?
      `;
      const rows = db.prepare(sql).all(buf, candidateK, filter.repo, k) as Array<{
        repo: string;
        file_path: string;
        page_content: string;
        metadata: string;
        distance: number;
      }>;
      return rows.map((r) => ({
        pageContent: r.page_content,
        metadata: JSON.parse(r.metadata) as Record<string, unknown>,
        distance: r.distance,
      }));
    }

    const sql = `
      SELECT d.page_content AS page_content,
             d.metadata AS metadata,
             m.distance AS distance
      FROM (SELECT rowid, distance FROM vectors WHERE embedding MATCH ? AND k = ?) m
      JOIN docs d ON d.rowid = m.rowid
      ORDER BY m.distance
      LIMIT ?
    `;
    const rows = db.prepare(sql).all(buf, candidateK, k) as Array<{
      page_content: string;
      metadata: string;
      distance: number;
    }>;
    return rows.map((r) => ({
      pageContent: r.page_content,
      metadata: JSON.parse(r.metadata) as Record<string, unknown>,
      distance: r.distance,
    }));
  }

  function insertEntry(entry: StoredEntry): void {
    const metadata = entry.metadata;
    const repo = extractString(metadata, "repo");
    const filePath = extractString(metadata, "filePath");
    if (!repo || !filePath) {
      throw new Error("Stored entries must carry string repo and filePath in metadata.");
    }
    const fileHash = extractString(metadata, "fileHash");
    const metaJson = JSON.stringify(metadata);
    const result = stmts.insertDoc.run(
      repo,
      filePath,
      entry.pageContent,
      metaJson,
      fileHash,
    );
    vec().insertVec.run(BigInt(result.lastInsertRowid), toFloat32Buffer(entry.embedding));
  }

  function insertBatchInternal(entries: StoredEntry[]): void {
    if (entries.length === 0) return;
    const tx = db.transaction(() => {
      for (const entry of entries) insertEntry(entry);
      bumpEpochInternal();
    });
    tx();
  }

  function upsertFileInternal(
    repo: string,
    filePath: string,
    _fileHash: string | null,
    entries: StoredEntry[],
  ): { added: number; removed: number } {
    let removed = 0;
    const tx = db.transaction(() => {
      vec().deleteVecByFile.run(repo, filePath);
      const deleted = stmts.deleteDocsByFile.all(repo, filePath) as Array<{ rowid: number }>;
      removed = deleted.length;
      for (const entry of entries) {
        // Ensure metadata mirrors what upsertFile received so the stored row
        // matches what the caller thought it wrote.
        const metadata = { ...entry.metadata, repo, filePath };
        insertEntry({ ...entry, metadata });
      }
      bumpEpochInternal();
    });
    tx();
    return { added: entries.length, removed };
  }

  function deleteByFileInternal(repo: string, filePath: string): number {
    let removed = 0;
    const tx = db.transaction(() => {
      vec().deleteVecByFile.run(repo, filePath);
      const deleted = stmts.deleteDocsByFile.all(repo, filePath) as Array<{ rowid: number }>;
      removed = deleted.length;
      if (removed > 0) bumpEpochInternal();
    });
    tx();
    return removed;
  }

  function deleteByRepoInternal(repo: string): number {
    let removed = 0;
    const tx = db.transaction(() => {
      vec().deleteVecByRepo.run(repo);
      const deleted = stmts.deleteDocsByRepo.all(repo) as Array<{ rowid: number }>;
      removed = deleted.length;
      if (removed > 0) bumpEpochInternal();
    });
    tx();
    return removed;
  }

  function fileSignaturesInternal(): Map<string, FileSignature> {
    const rows = stmts.selectFileSignatures.all() as Array<{
      repo: string;
      filePath: string;
      fileHash: string | null;
    }>;
    const map = new Map<string, FileSignature>();
    for (const row of rows) {
      const key = fileKey(row.repo, row.filePath);
      const existing = map.get(key);
      if (!existing) {
        map.set(key, { repo: row.repo, filePath: row.filePath, fileHash: row.fileHash });
        continue;
      }
      // If any row for this file has a null hash or differs, treat the whole
      // file as dirty by dropping the shared hash. This mirrors the JSONL
      // groupVectorsByFile behavior.
      if (existing.fileHash !== row.fileHash) {
        map.set(key, { ...existing, fileHash: null });
      }
    }
    return map;
  }

  function bumpEpochInternal(): void {
    const current = stmts.selectEpoch.get() as { value: string } | undefined;
    const next = (Number(current?.value ?? 0) + 1).toString();
    stmts.upsertEpoch.run(next);
  }

  function getWriteEpochInternal(): number {
    const current = stmts.selectEpoch.get() as { value: string } | undefined;
    return Number(current?.value ?? 0);
  }

  return {
    dbPath,
    getMeta: getMetaInternal,
    initializeSchema: initializeSchemaInternal,
    assertCompatibleWithConfig: assertCompatibleWithConfigInternal,
    count: () => (stmts.countDocs.get() as { c: number }).c,
    listRepos: listReposInternal,
    similaritySearch: similaritySearchInternal,
    upsertFile: upsertFileInternal,
    deleteByFile: deleteByFileInternal,
    deleteByRepo: deleteByRepoInternal,
    fileSignatures: fileSignaturesInternal,
    insertBatch: insertBatchInternal,
    bumpWriteEpoch: bumpEpochInternal,
    getWriteEpoch: getWriteEpochInternal,
    close: () => db.close(),
  };
}
