import { readdir, stat, readFile } from "node:fs/promises";
import { join, relative, extname } from "node:path";
import { createHash } from "node:crypto";
import { DEFAULT_SKIP_DIRS } from "./skip-dirs.js";

// Sentinel filename. Any directory containing this file (anywhere in the
// scan tree) is treated as "do not index this subtree". Documented in
// docs/configuration.md.
export const SKIP_SENTINEL = ".codebase-oracle-skip";

export const DEFAULT_INCLUDE_EXTENSIONS: ReadonlySet<string> = new Set([
  // JS / TS ecosystem
  ".ts", ".tsx", ".js", ".jsx", ".md", ".prisma", ".json",
  // Common sibling languages across the indexed repos
  ".py", ".php", ".go", ".rs", ".java", ".vue",
  // Config / infra / scripts
  ".yaml", ".yml", ".sh", ".toml", ".sql",
]);

const JSON_ALLOWLIST = new Set([
  "package.json", "tsconfig.json",
]);

// Fallback when neither config nor WalkRepoOptions supplies a limit.
// Config.maxFileSizeBytes (config.ts) defaults to this same value, so in
// practice callers that go through loadConfig() always pass an explicit
// number; this constant only matters for direct walkRepo() callers (tests).
export const DEFAULT_MAX_FILE_SIZE_BYTES = 500_000;

export interface ScannedFile {
  absolutePath: string;
  relativePath: string; // relative to scanRoot
  repo: string;         // repo directory name
  language: string;     // extension without dot
  content: string;
  contentHash: string;
}

export interface RepoInfo {
  name: string;
  path: string;
  fileCount: number;
}

export async function discoverRepos(scanRoot: string): Promise<RepoInfo[]> {
  const entries = await readdir(scanRoot, { withFileTypes: true });
  const repos: RepoInfo[] = [];

  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    if (entry.name.startsWith(".")) continue;
    // Check if it's a git repo (has .git dir or is a file like a submodule)
    const gitPath = join(scanRoot, entry.name, ".git");
    try {
      await stat(gitPath);
      repos.push({ name: entry.name, path: join(scanRoot, entry.name), fileCount: 0 });
    } catch {
      // Not a git repo, skip
    }
  }

  return repos.sort((a, b) => a.name.localeCompare(b.name));
}

// Reported through onSkip whenever walkRepo declines to yield a file it
// otherwise would have (i.e. everything except the silent "empty file"
// case, which has nothing worth indexing and so stays quiet — see the
// comment at the empty-file check below).
export interface SkippedFile {
  repo: string;
  relativePath: string;
  absolutePath: string;
  reason: "too-large" | "read-error";
  sizeBytes?: number;
  limitBytes?: number;
  message?: string;
}

export interface WalkRepoOptions {
  extensions?: ReadonlySet<string>;
  skipDirs?: ReadonlySet<string>;
  maxFileSizeBytes?: number;
  /** Per-file skip reporter. Defaults to a no-op — callers that care (the
   * indexer, watch mode) pass one so a skip is always visible somewhere,
   * never just a file that quietly never shows up in the index. */
  onSkip?: (skip: SkippedFile) => void;
}

export async function* walkRepo(
  repoPath: string,
  repoName: string,
  scanRoot: string,
  options?: WalkRepoOptions,
): AsyncGenerator<ScannedFile> {
  const extensions = options?.extensions ?? DEFAULT_INCLUDE_EXTENSIONS;
  const skipDirs = options?.skipDirs ?? DEFAULT_SKIP_DIRS;
  const maxFileSizeBytes = options?.maxFileSizeBytes ?? DEFAULT_MAX_FILE_SIZE_BYTES;
  const onSkip = options?.onSkip ?? (() => {});
  // Lockfiles + per-package manifests explode the index, so we only whitelist
  // a couple by name when the user hasn't taken control of the extension list.
  // An explicit override means the user knows what they're asking for.
  const applyJsonAllowlist = !options?.extensions;

  async function* walk(dir: string): AsyncGenerator<ScannedFile> {
    const entries = await readdir(dir, { withFileTypes: true });

    // Tree-level opt-out. A directory that contains a `.codebase-oracle-skip`
    // sentinel file is pruned wholesale, regardless of name. Lets vendored
    // fixtures (tests/eval/corpus/) and other "documentation that lives in
    // the source tree but should not enter the index" subtrees stay
    // co-located with the code that owns them without polluting queries.
    if (entries.some((e) => e.name === SKIP_SENTINEL && e.isFile())) {
      return;
    }

    for (const entry of entries) {
      const fullPath = join(dir, entry.name);

      if (entry.isDirectory()) {
        if (skipDirs.has(entry.name)) continue;
        yield* walk(fullPath);
        continue;
      }

      if (!entry.isFile()) continue;

      const ext = extname(entry.name);
      if (!extensions.has(ext)) continue;

      if (ext === ".json" && applyJsonAllowlist && !JSON_ALLOWLIST.has(entry.name)) continue;

      const relativePath = relative(scanRoot, fullPath);

      try {
        // Stat before read: a size check in true bytes, decided BEFORE we
        // pull the file into memory. Reading a multi-MB file into a string
        // just to discard it (the old content.length > 200_000 check, which
        // also mismeasured UTF-16 chars as bytes) wastes memory and time on
        // every over-limit file, every run.
        const st = await stat(fullPath);
        if (st.size > maxFileSizeBytes) {
          onSkip({
            repo: repoName,
            relativePath,
            absolutePath: fullPath,
            reason: "too-large",
            sizeBytes: st.size,
            limitBytes: maxFileSizeBytes,
          });
          continue;
        }

        const content = await readFile(fullPath, "utf-8");
        // Empty files have nothing worth indexing, so this one stays a
        // silent skip on purpose — unlike too-large/read-error, it is not
        // an anomaly worth a WARNING line.
        if (!content.trim()) continue;

        yield {
          absolutePath: fullPath,
          relativePath,
          repo: repoName,
          language: ext.slice(1),
          content,
          contentHash: createHash("sha256").update(content).digest("hex"),
        };
      } catch (err) {
        // Permission error, binary decode failure, or a stat/read race
        // (file removed between readdir and stat). Reported, never
        // swallowed — but still non-throwing: one bad file must not kill
        // the whole scan.
        onSkip({
          repo: repoName,
          relativePath,
          absolutePath: fullPath,
          reason: "read-error",
          message: err instanceof Error ? err.message : String(err),
        });
      }
    }
  }

  yield* walk(repoPath);
}
