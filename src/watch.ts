import chokidar from "chokidar";
import { basename, dirname, extname, join, relative, sep } from "node:path";
import { readFile, stat } from "node:fs/promises";
import { createHash } from "node:crypto";
import { EventEmitter } from "node:events";
import type { Embeddings } from "@langchain/core/embeddings";
import { assertScanRoot, type Config } from "./config.js";
import { createEmbeddings } from "./store/embeddings.js";
import {
  DEFAULT_INCLUDE_EXTENSIONS,
  DEFAULT_MAX_FILE_SIZE_BYTES,
  discoverRepos,
  type ScannedFile,
} from "./ingest/scanner.js";
import { mergeSkipDirs } from "./ingest/skip-dirs.js";
import { splitFile } from "./ingest/splitter.js";
import {
  openSqliteStore,
  type SqliteStore,
  type StoredEntry,
} from "./store/sqlite-store.js";
const JSON_ALLOWLIST = new Set(["package.json", "tsconfig.json"]);
const DEFAULT_DEBOUNCE_MS = 3000;

export interface RunningWatcher {
  close: () => Promise<void>;
  stats: () => { vectors: number; repos: number; pending: number };
  /** Resolves after the next scheduled flush completes. Test hook. */
  flushOnce: () => Promise<void>;
  /**
   * Resolves as soon as the accumulated-but-not-yet-flushed event count
   * reaches minCount. Event-driven (subscribes to the internal pending-count
   * signal) rather than polling on a fixed cadence, so it settles the
   * instant chokidar's awaitWriteFinish/debounce actually fires instead of
   * racing a wall-clock budget against filesystem-event latency under load.
   * timeoutMs is a generous backstop for a genuinely broken watcher, not a
   * tightly-tuned deadline for the happy path. Test hook.
   */
  waitForPending: (minCount: number, timeoutMs?: number) => Promise<void>;
}

export interface WatchOptions {
  debounceMs?: number;
  /** Inject a fake Embeddings implementation (tests). Defaults to createEmbeddings(config). */
  embeddings?: Embeddings;
  /** Inject a store handle (tests). Defaults to openSqliteStore(config). */
  store?: SqliteStore;
}

/** Pure helper: does this filename match the active extension filter? */
export function shouldIndexPath(
  fileName: string,
  extensions: ReadonlySet<string>,
  applyJsonAllowlist: boolean,
): boolean {
  const ext = extname(fileName);
  if (!extensions.has(ext)) return false;
  if (ext === ".json" && applyJsonAllowlist && !JSON_ALLOWLIST.has(fileName)) return false;
  return true;
}

/** Pure helper: given an absolute path, identify the (repo, relativePath). */
export function computeRepoAndRelativePath(
  absolutePath: string,
  scanRoot: string,
  knownRepos: ReadonlyMap<string, string>,
): { repo: string; relativePath: string } | null {
  const rel = relative(scanRoot, absolutePath);
  if (rel === "" || rel.startsWith("..")) return null;
  const first = rel.split(sep)[0];
  const repoAbsRoot = join(scanRoot, first);
  const repoName = knownRepos.get(repoAbsRoot);
  if (!repoName) return null;
  return { repo: repoName, relativePath: rel };
}

interface FileEvent {
  kind: "upsert" | "delete";
  absolutePath: string;
  repo: string;
  relativePath: string;
}

/**
 * Accumulator that dedups events per (repo, relativePath) until drained.
 * Semantics: latest file event wins. Recording a repo-deletion drops any
 * pending per-file events for that repo.
 */
export class PendingEventMap {
  private files = new Map<string, FileEvent>();
  private repos = new Set<string>();

  recordUpsert(absolutePath: string, repo: string, relativePath: string): void {
    const key = `${repo}::${relativePath}`;
    this.files.set(key, { kind: "upsert", absolutePath, repo, relativePath });
  }

  recordDelete(repo: string, relativePath: string): void {
    const key = `${repo}::${relativePath}`;
    this.files.set(key, { kind: "delete", absolutePath: "", repo, relativePath });
  }

  recordRepoDelete(repo: string): void {
    this.repos.add(repo);
    for (const [key, ev] of this.files) {
      if (ev.repo === repo) this.files.delete(key);
    }
  }

  size(): number {
    return this.files.size + this.repos.size;
  }

  drain(): { files: FileEvent[]; repos: string[] } {
    const drained = { files: [...this.files.values()], repos: [...this.repos] };
    this.files.clear();
    this.repos.clear();
    return drained;
  }
}

// Result of trying to load a changed file for re-embedding. Distinguishes
// "too-large" (an anomaly worth a loud WARNING — the file used to be
// indexed, or should be, and now silently wouldn't be) from "empty" (nothing
// to index, unremarkable, mirrors the silent-skip in scanner.ts) so the
// caller can log accordingly.
type LoadResult =
  | { kind: "ok"; file: ScannedFile }
  | { kind: "empty" }
  | { kind: "too-large"; sizeBytes: number; limitBytes: number };

async function loadScannedFile(
  absolutePath: string,
  relativePath: string,
  repo: string,
  maxFileSizeBytes: number | undefined,
): Promise<LoadResult> {
  // Mirrors walkRepo's fallback (scanner.ts). Config.maxFileSizeBytes is
  // required after loadConfig, but tests hand-build Config literals and are
  // not typechecked (tsconfig only includes src/) — an undefined limit here
  // would make `st.size > undefined` always false and silently re-open the
  // very drop this feature closes.
  const limit = maxFileSizeBytes ?? DEFAULT_MAX_FILE_SIZE_BYTES;
  // Stat-first in true bytes, same reasoning as scanner.ts: decide before
  // reading the file into memory, and measure real bytes rather than
  // UTF-16 string length.
  const st = await stat(absolutePath);
  if (st.size > limit) {
    return { kind: "too-large", sizeBytes: st.size, limitBytes: limit };
  }
  const content = await readFile(absolutePath, "utf-8");
  if (!content.trim()) return { kind: "empty" };
  const ext = extname(absolutePath);
  return {
    kind: "ok",
    file: {
      absolutePath,
      relativePath,
      repo,
      language: ext.slice(1),
      content,
      contentHash: createHash("sha256").update(content).digest("hex"),
    },
  };
}

async function embedFile(
  embeddings: Embeddings,
  scanned: ScannedFile,
): Promise<StoredEntry[]> {
  const chunks = await splitFile(scanned);
  if (chunks.length === 0) return [];
  const texts = chunks.map((c) => c.pageContent);
  const embs = await embeddings.embedDocuments(texts);
  return chunks.map((chunk, i) => ({
    embedding: embs[i],
    pageContent: chunk.pageContent,
    metadata: chunk.metadata as Record<string, unknown>,
  }));
}

export async function runWatchMode(
  config: Config,
  options: WatchOptions = {},
): Promise<RunningWatcher> {
  assertScanRoot(config);
  const debounceMs = options.debounceMs ?? DEFAULT_DEBOUNCE_MS;

  const store = options.store ?? openSqliteStore(config);
  const ownedStore = !options.store;
  store.assertCompatibleWithConfig(config);

  const repos = await discoverRepos(config.scanRoot);
  const repoRoots = new Map<string, string>(repos.map((r) => [r.path, r.name]));

  const extensions = config.includeExtensions
    ? new Set(config.includeExtensions)
    : DEFAULT_INCLUDE_EXTENSIONS;
  const applyJsonAllowlist = !config.includeExtensions;
  const skipDirs = mergeSkipDirs(config.skipDirs);
  const embeddings = options.embeddings ?? createEmbeddings(config);
  const pending = new PendingEventMap();
  // Fires whenever an event is enqueued into `pending` (not on drain), so
  // waitForPending below can await the real signal instead of polling.
  const pendingEvents = new EventEmitter();

  const initialCount = store.count();
  const meta = store.getMeta();
  let expectedDim: number | null = meta?.dimension ?? null;

  console.log(
    `watch: scanRoot=${config.scanRoot} repos=${repos.length} debounce=${debounceMs}ms ` +
      `vectors=${initialCount}`,
  );

  let timer: NodeJS.Timeout | null = null;
  let processing = false;
  let inflight: Promise<void> | null = null;

  const ensureDimInitialized = (dim: number) => {
    if (expectedDim === null) {
      store.initializeSchema({
        embeddingProvider: config.embeddingProvider,
        embeddingModel: config.embeddingModel,
        dimension: dim,
      });
      expectedDim = dim;
    }
  };

  const scheduleFlush = () => {
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => {
      timer = null;
      // If a flush is already running, don't start another — its `finally`
      // block reschedules us if the pending queue has grown.
      if (processing) return;
      inflight = flush().finally(() => {
        inflight = null;
      });
    }, debounceMs);
  };

  const flush = async () => {
    if (processing) return; // defensive; scheduleFlush should have prevented this.
    processing = true;
    try {
      const { files, repos: droppedRepos } = pending.drain();
      if (files.length === 0 && droppedRepos.length === 0) return;

      for (const repo of droppedRepos) {
        const removed = store.deleteByRepo(repo);
        if (removed > 0) {
          console.log(`watch: repo ${repo} gone (-${removed} vectors)`);
        }
      }

      for (const ev of files) {
        if (ev.kind === "delete") {
          const removed = store.deleteByFile(ev.repo, ev.relativePath);
          if (removed > 0) {
            console.log(`watch: removed ${ev.relativePath} (-${removed} chunks)`);
          }
          continue;
        }

        // Upsert: compute new vectors FIRST, only swap on success. An embed
        // failure leaves the old vectors in place instead of net-losing them.
        let loaded: LoadResult;
        try {
          loaded = await loadScannedFile(
            ev.absolutePath,
            ev.relativePath,
            ev.repo,
            config.maxFileSizeBytes,
          );
        } catch (err) {
          console.warn(
            `watch: failed to read ${ev.relativePath}:`,
            err instanceof Error ? err.message : err,
          );
          continue;
        }

        if (loaded.kind === "too-large") {
          console.warn(
            `WARNING: skipped ${ev.relativePath} — ${loaded.sizeBytes} bytes > `
              + `ORACLE_MAX_FILE_SIZE=${loaded.limitBytes}`,
          );
          const removed = store.deleteByFile(ev.repo, ev.relativePath);
          if (removed > 0) {
            console.log(`watch: ${ev.relativePath} unindexed (too large) (-${removed} chunks)`);
          }
          continue;
        }

        if (loaded.kind === "empty") {
          // Empty files have nothing worth indexing — silent skip on
          // purpose, mirroring scanner.ts. Any stale vectors from a
          // previously non-empty version of the file are still cleared.
          const removed = store.deleteByFile(ev.repo, ev.relativePath);
          if (removed > 0) {
            console.log(`watch: ${ev.relativePath} unindexed (empty) (-${removed} chunks)`);
          }
          continue;
        }

        const scanned = loaded.file;

        let newEntries: StoredEntry[];
        try {
          newEntries = await embedFile(embeddings, scanned);
        } catch (err) {
          console.warn(
            `watch: failed to embed ${ev.relativePath}:`,
            err instanceof Error ? err.message : err,
          );
          continue;
        }

        if (newEntries.length > 0) {
          ensureDimInitialized(newEntries[0].embedding.length);
        }

        const { added, removed } = store.upsertFile(
          ev.repo,
          ev.relativePath,
          scanned.contentHash,
          newEntries,
        );
        console.log(
          `watch: reembedded ${ev.relativePath} (+${added} chunks, -${removed} chunks)`,
        );
      }
    } finally {
      processing = false;
      if (pending.size() > 0) scheduleFlush();
    }
  };

  const enqueueFileEvent = (absolutePath: string, kind: "upsert" | "delete") => {
    const parts = absolutePath.split(sep);
    const fileName = parts[parts.length - 1];
    if (!shouldIndexPath(fileName, extensions, applyJsonAllowlist)) return;
    const resolved = computeRepoAndRelativePath(absolutePath, config.scanRoot, repoRoots);
    if (!resolved) return;
    if (kind === "upsert") {
      pending.recordUpsert(absolutePath, resolved.repo, resolved.relativePath);
    } else {
      pending.recordDelete(resolved.repo, resolved.relativePath);
    }
    pendingEvents.emit("change", pending.size());
    scheduleFlush();
  };

  const watcher = chokidar.watch(config.scanRoot, {
    ignored: (p) => {
      const parts = p.split(sep);
      return parts.some((part) => skipDirs.has(part));
    },
    ignoreInitial: true,
    persistent: true,
    awaitWriteFinish: { stabilityThreshold: 500, pollInterval: 100 },
  });

  watcher.on("add", (p) => enqueueFileEvent(p, "upsert"));
  watcher.on("change", (p) => enqueueFileEvent(p, "upsert"));
  watcher.on("unlink", (p) => enqueueFileEvent(p, "delete"));

  watcher.on("unlinkDir", (absDir) => {
    const repo = repoRoots.get(absDir);
    if (!repo) return;
    repoRoots.delete(absDir);
    pending.recordRepoDelete(repo);
    pendingEvents.emit("change", pending.size());
    scheduleFlush();
  });

  watcher.on("addDir", async (absDir) => {
    // `.git` is ignored, so we never see it directly. Instead, watch for
    // top-level dirs under scanRoot and check whether they contain a .git.
    if (repoRoots.has(absDir)) return;
    if (dirname(absDir) !== config.scanRoot) return;
    try {
      await stat(join(absDir, ".git"));
    } catch {
      return; // not a git repo (yet).
    }
    const name = basename(absDir);
    if (!name) return;
    repoRoots.set(absDir, name);
    console.log(
      `watch: new repo "${name}" detected. Run 'npm run index' to back-fill its existing files; ` +
        `subsequent changes will be picked up incrementally.`,
    );
  });

  watcher.on("error", (err) => {
    console.error("watch: chokidar error:", err instanceof Error ? err.message : err);
  });

  await new Promise<void>((resolve) => watcher.once("ready", () => resolve()));
  console.log("watch: ready. Ctrl+C to exit.");

  const waitForPending = (minCount: number, timeoutMs = 20_000): Promise<void> => {
    if (pending.size() >= minCount) return Promise.resolve();
    return new Promise<void>((resolve, reject) => {
      let timeoutTimer: NodeJS.Timeout;
      const onChange = (size: number) => {
        if (size < minCount) return;
        clearTimeout(timeoutTimer);
        pendingEvents.off("change", onChange);
        resolve();
      };
      timeoutTimer = setTimeout(() => {
        pendingEvents.off("change", onChange);
        reject(
          new Error(
            `timed out waiting for pending events (expected >= ${minCount}, saw ${pending.size()})`,
          ),
        );
      }, timeoutMs);
      pendingEvents.on("change", onChange);
    });
  };

  return {
    close: async () => {
      if (timer) {
        clearTimeout(timer);
        timer = null;
      }
      if (inflight) await inflight;
      await watcher.close();
      if (ownedStore) store.close();
    },
    stats: () => ({
      vectors: store.count(),
      repos: repoRoots.size,
      pending: pending.size(),
    }),
    flushOnce: async () => {
      if (timer) {
        clearTimeout(timer);
        timer = null;
      }
      await flush();
    },
    waitForPending,
  };
}
