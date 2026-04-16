import { readdir, stat, readFile } from "node:fs/promises";
import { join, relative, extname } from "node:path";
import { createHash } from "node:crypto";

const SKIP_DIRS = new Set([
  "node_modules", ".git", "dist", "build", ".next", ".turbo",
  "coverage", ".nyc_output", "__pycache__", ".venv", "vendor",
]);

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

export interface WalkRepoOptions {
  extensions?: ReadonlySet<string>;
}

export async function* walkRepo(
  repoPath: string,
  repoName: string,
  scanRoot: string,
  options?: WalkRepoOptions,
): AsyncGenerator<ScannedFile> {
  const extensions = options?.extensions ?? DEFAULT_INCLUDE_EXTENSIONS;
  // Lockfiles + per-package manifests explode the index, so we only whitelist
  // a couple by name when the user hasn't taken control of the extension list.
  // An explicit override means the user knows what they're asking for.
  const applyJsonAllowlist = !options?.extensions;

  async function* walk(dir: string): AsyncGenerator<ScannedFile> {
    const entries = await readdir(dir, { withFileTypes: true });

    for (const entry of entries) {
      const fullPath = join(dir, entry.name);

      if (entry.isDirectory()) {
        if (SKIP_DIRS.has(entry.name)) continue;
        yield* walk(fullPath);
        continue;
      }

      if (!entry.isFile()) continue;

      const ext = extname(entry.name);
      if (!extensions.has(ext)) continue;

      if (ext === ".json" && applyJsonAllowlist && !JSON_ALLOWLIST.has(entry.name)) continue;

      try {
        const content = await readFile(fullPath, "utf-8");
        // Skip empty or very large files
        if (!content.trim() || content.length > 200_000) continue;

        yield {
          absolutePath: fullPath,
          relativePath: relative(scanRoot, fullPath),
          repo: repoName,
          language: ext.slice(1),
          content,
          contentHash: createHash("sha256").update(content).digest("hex"),
        };
      } catch {
        // Permission error or binary, skip
      }
    }
  }

  yield* walk(repoPath);
}
