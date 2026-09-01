/**
 * Helpers for rendering an `IndexedRepo`'s `lastIndexedAt` timestamp in a
 * way agents can act on at a glance: relative ("12 min ago", "3 days ago")
 * plus an explicit cutoff for "never indexed under the freshness rollout".
 */

const SECOND = 1_000;
const MINUTE = 60 * SECOND;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;

export function formatRelativeFreshness(
  isoTimestamp: string | null,
  now: Date = new Date(),
): string {
  if (!isoTimestamp) return "never";
  const then = new Date(isoTimestamp);
  if (Number.isNaN(then.getTime())) return "never";
  const diffMs = now.getTime() - then.getTime();
  if (diffMs < 0) return "just now";
  if (diffMs < MINUTE) return "just now";
  if (diffMs < HOUR) {
    const m = Math.floor(diffMs / MINUTE);
    return `${m} min ago`;
  }
  if (diffMs < DAY) {
    const h = Math.floor(diffMs / HOUR);
    return `${h} hour${h === 1 ? "" : "s"} ago`;
  }
  const d = Math.floor(diffMs / DAY);
  return `${d} day${d === 1 ? "" : "s"} ago`;
}

/**
 * Render a single repo line for `oracle_list_repos` output. Includes the
 * indexedAt suffix only when a timestamp is present so legacy stores keep
 * showing the bare `<repo> — N chunks across M files` form. Appends a
 * `skipped` suffix only when the last index run actually skipped a file for
 * this repo (both `skippedSizeCount` and `skippedErrorCount` omitted or 0
 * renders nothing extra) — this is what surfaces the size-ceiling skip
 * count from repo_skip_meta on the CLI, MCP, and HTTP list-repos surfaces,
 * which all share this function.
 *
 * `prefix` lets callers pick the leading marker (default `"- "` for the
 * MCP/HTTP markdown-ish output; the CLI passes `"  "` for plain indent).
 */
export function formatRepoLine(
  repo: {
    repo: string;
    chunkCount: number;
    fileCount: number;
    lastIndexedAt: string | null;
    skippedSizeCount?: number;
    skippedErrorCount?: number;
  },
  optionsOrNow?: Date | { now?: Date; prefix?: string },
): string {
  const opts =
    optionsOrNow instanceof Date ? { now: optionsOrNow } : optionsOrNow ?? {};
  const prefix = opts.prefix ?? "- ";
  let line = `${prefix}${repo.repo} — ${repo.chunkCount} chunks across ${repo.fileCount} files`;
  if (repo.lastIndexedAt) {
    const relative = formatRelativeFreshness(repo.lastIndexedAt, opts.now);
    line += ` (indexed ${repo.lastIndexedAt}, ${relative})`;
  }
  const skippedTotal = (repo.skippedSizeCount ?? 0) + (repo.skippedErrorCount ?? 0);
  if (skippedTotal > 0) {
    line += `; ${skippedTotal} file(s) skipped in the last index run`;
  }
  return line;
}
