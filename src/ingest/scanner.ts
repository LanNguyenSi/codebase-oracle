import { readdir, stat, readFile } from "node:fs/promises";
import { join, relative, extname } from "node:path";

const SKIP_DIRS = new Set([
  "node_modules", ".git", "dist", "build", ".next", ".turbo",
  "coverage", ".nyc_output", "__pycache__", ".venv", "vendor",
]);

const INCLUDE_EXTENSIONS = new Set([
  ".ts", ".tsx", ".js", ".jsx", ".md", ".prisma", ".json",
]);

const JSON_ALLOWLIST = new Set([
  "package.json", "tsconfig.json",
]);

export interface ScannedFile {
  absolutePath: string;
  relativePath: string; // relative to pandoraRoot
  repo: string;         // repo directory name
  language: string;     // extension without dot
  content: string;
}

export interface RepoInfo {
  name: string;
  path: string;
  fileCount: number;
}

export async function discoverRepos(pandoraRoot: string): Promise<RepoInfo[]> {
  const entries = await readdir(pandoraRoot, { withFileTypes: true });
  const repos: RepoInfo[] = [];

  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    if (entry.name.startsWith(".")) continue;
    // Check if it's a git repo (has .git dir or is a file like a submodule)
    const gitPath = join(pandoraRoot, entry.name, ".git");
    try {
      await stat(gitPath);
      repos.push({ name: entry.name, path: join(pandoraRoot, entry.name), fileCount: 0 });
    } catch {
      // Not a git repo, skip
    }
  }

  return repos.sort((a, b) => a.name.localeCompare(b.name));
}

export async function* walkRepo(
  repoPath: string,
  repoName: string,
  pandoraRoot: string,
): AsyncGenerator<ScannedFile> {
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
      if (!INCLUDE_EXTENSIONS.has(ext)) continue;

      // Only allow specific JSON files
      if (ext === ".json" && !JSON_ALLOWLIST.has(entry.name)) continue;

      try {
        const content = await readFile(fullPath, "utf-8");
        // Skip empty or very large files
        if (!content.trim() || content.length > 200_000) continue;

        yield {
          absolutePath: fullPath,
          relativePath: relative(pandoraRoot, fullPath),
          repo: repoName,
          language: ext.slice(1),
          content,
        };
      } catch {
        // Permission error or binary, skip
      }
    }
  }

  yield* walk(repoPath);
}
