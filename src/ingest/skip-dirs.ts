// Directory names the scanner and watcher must never descend into.
//
// Matching is per-segment exact-name. A directory hit anywhere in a path
// prunes that subtree entirely — so `.bun` here means any `.bun` directory
// at any depth gets skipped, which is enough to cover vendored package
// caches like `.opencode-home/.bun/install/cache/`.
//
// Adding new entries here is the safe way to extend defaults. End users
// can also pass extra names via `ORACLE_SKIP_DIRS` (see config.ts).
export const DEFAULT_SKIP_DIRS: ReadonlySet<string> = new Set([
  // VCS / language runtimes
  ".git",
  "__pycache__",
  ".venv",

  // Build / cache output
  "build",
  "coverage",
  "dist",
  ".cache",
  ".next",
  ".nyc_output",
  ".turbo",

  // Package managers and their stores
  "node_modules",
  ".bun",
  ".pnpm-store",
  ".yarn",
  "vendor",

  // Vendored agent / IDE workspaces that ship third-party caches
  ".husky",
  ".idea",
  ".opencode-home",
  ".vscode",
]);

// Returns a new Set that contains every default plus the caller-supplied
// extras. Used by the CLI to honour `ORACLE_SKIP_DIRS` without letting the
// override drop critical defaults (forgetting `node_modules` would explode
// the index).
export function mergeSkipDirs(
  extras: readonly string[] | undefined,
): ReadonlySet<string> {
  if (!extras || extras.length === 0) return DEFAULT_SKIP_DIRS;
  const merged = new Set(DEFAULT_SKIP_DIRS);
  for (const name of extras) {
    const trimmed = name.trim();
    if (trimmed) merged.add(trimmed);
  }
  return merged;
}
