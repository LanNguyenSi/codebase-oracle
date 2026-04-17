import chokidar from "chokidar";
import { basename, dirname, extname, join, relative, sep } from "node:path";
import { readFile, stat } from "node:fs/promises";
import { createHash } from "node:crypto";
import type { Embeddings } from "@langchain/core/embeddings";
import type { Config } from "./config.js";
import { createEmbeddings } from "./store/embeddings.js";
import {
  DEFAULT_INCLUDE_EXTENSIONS,
  discoverRepos,
  type ScannedFile,
} from "./ingest/scanner.js";
import { splitFile } from "./ingest/splitter.js";
import {
  assertCompatibleIndex,
  loadStoredVectorsWithMeta,
  persistStoredVectors,
  type StoredVector,
} from "./store/vector-store.js";

const WATCH_SKIP_DIRS = new Set([
  "node_modules",
  ".git",
  "dist",
  "build",
  ".next",
  ".turbo",
  "coverage",
  ".nyc_output",
  "__pycache__",
  ".venv",
  "vendor",
]);
const JSON_ALLOWLIST = new Set(["package.json", "tsconfig.json"]);
const MAX_FILE_BYTES = 200_000;
const DEFAULT_DEBOUNCE_MS = 3000;

export interface RunningWatcher {
  close: () => Promise<void>;
  stats: () => { vectors: number; repos: number; pending: number };
  /** Resolves after the next scheduled flush completes. Test hook. */
  flushOnce: () => Promise<void>;
}

export interface WatchOptions {
  debounceMs?: number;
  /** Inject a fake Embeddings implementation (tests). Defaults to createEmbeddings(config). */
  embeddings?: Embeddings;
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

async function loadScannedFile(
  absolutePath: string,
  relativePath: string,
  repo: string,
): Promise<ScannedFile | null> {
  const content = await readFile(absolutePath, "utf-8");
  if (!content.trim() || content.length > MAX_FILE_BYTES) return null;
  const ext = extname(absolutePath);
  return {
    absolutePath,
    relativePath,
    repo,
    language: ext.slice(1),
    content,
    contentHash: createHash("sha256").update(content).digest("hex"),
  };
}

function removeVectorsForFile(
  vectors: StoredVector[],
  repo: string,
  relativePath: string,
): number {
  const before = vectors.length;
  for (let i = vectors.length - 1; i >= 0; i--) {
    const m = vectors[i].doc.metadata as { repo?: string; filePath?: string };
    if (m.repo === repo && m.filePath === relativePath) vectors.splice(i, 1);
  }
  return before - vectors.length;
}

function removeVectorsForRepo(vectors: StoredVector[], repo: string): number {
  const before = vectors.length;
  for (let i = vectors.length - 1; i >= 0; i--) {
    const m = vectors[i].doc.metadata as { repo?: string };
    if (m.repo === repo) vectors.splice(i, 1);
  }
  return before - vectors.length;
}

async function embedFile(
  embeddings: Embeddings,
  scanned: ScannedFile,
): Promise<StoredVector[]> {
  const chunks = await splitFile(scanned);
  if (chunks.length === 0) return [];
  const texts = chunks.map((c) => c.pageContent);
  const embs = await embeddings.embedDocuments(texts);
  return chunks.map((chunk, i) => ({
    embedding: embs[i],
    doc: {
      pageContent: chunk.pageContent,
      metadata: chunk.metadata as Record<string, unknown>,
    },
  }));
}

export async function runWatchMode(
  config: Config,
  options: WatchOptions = {},
): Promise<RunningWatcher> {
  const debounceMs = options.debounceMs ?? DEFAULT_DEBOUNCE_MS;

  const { vectors: initialVectors, meta } = await loadStoredVectorsWithMeta(config);
  assertCompatibleIndex(initialVectors, meta, config);
  const vectors: StoredVector[] = [...initialVectors];

  const repos = await discoverRepos(config.scanRoot);
  const repoRoots = new Map<string, string>(repos.map((r) => [r.path, r.name]));

  const extensions = config.includeExtensions
    ? new Set(config.includeExtensions)
    : DEFAULT_INCLUDE_EXTENSIONS;
  const applyJsonAllowlist = !config.includeExtensions;
  const embeddings = options.embeddings ?? createEmbeddings(config);
  const pending = new PendingEventMap();

  console.log(
    `watch: scanRoot=${config.scanRoot} repos=${repos.length} debounce=${debounceMs}ms ` +
      `vectors=${vectors.length}`,
  );

  let timer: NodeJS.Timeout | null = null;
  let processing = false;
  let inflight: Promise<void> | null = null;

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

      let modified = false;

      for (const repo of droppedRepos) {
        const removed = removeVectorsForRepo(vectors, repo);
        if (removed > 0) {
          console.log(`watch: repo ${repo} gone (-${removed} vectors)`);
          modified = true;
        }
      }

      for (const ev of files) {
        if (ev.kind === "delete") {
          const removed = removeVectorsForFile(vectors, ev.repo, ev.relativePath);
          if (removed > 0) {
            console.log(`watch: removed ${ev.repo}/${ev.relativePath} (-${removed} chunks)`);
            modified = true;
          }
          continue;
        }

        // Upsert: compute new vectors FIRST, only swap on success. An embed
        // failure leaves the old vectors in place instead of net-losing them.
        let scanned: ScannedFile | null;
        try {
          scanned = await loadScannedFile(ev.absolutePath, ev.relativePath, ev.repo);
        } catch (err) {
          console.warn(
            `watch: failed to read ${ev.repo}/${ev.relativePath}:`,
            err instanceof Error ? err.message : err,
          );
          continue;
        }

        if (!scanned) {
          const removed = removeVectorsForFile(vectors, ev.repo, ev.relativePath);
          if (removed > 0) {
            console.log(
              `watch: ${ev.repo}/${ev.relativePath} unindexed (empty / too large) (-${removed} chunks)`,
            );
            modified = true;
          }
          continue;
        }

        let newVectors: StoredVector[];
        try {
          newVectors = await embedFile(embeddings, scanned);
        } catch (err) {
          console.warn(
            `watch: failed to embed ${ev.repo}/${ev.relativePath}:`,
            err instanceof Error ? err.message : err,
          );
          continue;
        }

        const removed = removeVectorsForFile(vectors, ev.repo, ev.relativePath);
        vectors.push(...newVectors);
        console.log(
          `watch: reembedded ${ev.repo}/${ev.relativePath} (+${newVectors.length} chunks, -${removed} chunks)`,
        );
        modified = true;
      }

      if (modified) {
        await persistStoredVectors(vectors, config);
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
    scheduleFlush();
  };

  const watcher = chokidar.watch(config.scanRoot, {
    ignored: (p) => {
      const parts = p.split(sep);
      return parts.some((part) => WATCH_SKIP_DIRS.has(part));
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

  return {
    close: async () => {
      if (timer) {
        clearTimeout(timer);
        timer = null;
      }
      if (inflight) await inflight;
      await watcher.close();
    },
    stats: () => ({
      vectors: vectors.length,
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
  };
}
