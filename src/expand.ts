import { readFile } from "node:fs/promises";
import type { VectorStoreWrapper } from "./store/vector-store.js";

export interface ExpandRequest {
  repo: string;
  path: string;
  /** 1-indexed line to center the window on. Defaults to 1 (top of file). */
  line?: number;
  /** Number of lines to include around `line` (above + below). Default 30, hard cap 200. */
  window?: number;
}

export interface ExpandSuccess {
  ok: true;
  repo: string;
  path: string;
  /** 1-indexed first line returned. */
  lineStart: number;
  /** 1-indexed last line returned, inclusive. */
  lineEnd: number;
  /** Total lines in the file at read time. */
  totalLines: number;
  /** cat-n style text: `   42→content` per line. */
  text: string;
}

export interface ExpandFailure {
  ok: false;
  reason:
    | "not_indexed"
    | "no_absolute_path"
    | "file_missing"
    | "read_error";
  message: string;
}

export type ExpandResult = ExpandSuccess | ExpandFailure;

const DEFAULT_WINDOW = 30;
const MAX_WINDOW = 200;

export async function expandFile(
  store: VectorStoreWrapper,
  req: ExpandRequest,
): Promise<ExpandResult> {
  const metadata = store.getFileMetadata(req.repo, req.path);
  if (!metadata) {
    return {
      ok: false,
      reason: "not_indexed",
      message: `No chunks indexed for ${req.repo}/${req.path}. Try oracle_search first to confirm the path, or oracle_list_repos to see indexed repos.`,
    };
  }

  const absolutePath = typeof metadata.absolutePath === "string" ? metadata.absolutePath : null;
  if (!absolutePath) {
    return {
      ok: false,
      reason: "no_absolute_path",
      message: `Indexed chunk for ${req.repo}/${req.path} has no absolutePath in metadata. Re-index the repo to populate it.`,
    };
  }

  let content: string;
  try {
    content = await readFile(absolutePath, "utf8");
  } catch (err) {
    const code = (err as NodeJS.ErrnoException).code;
    if (code === "ENOENT") {
      return {
        ok: false,
        reason: "file_missing",
        message: `Indexed file ${absolutePath} no longer exists on disk. The chunk you queried may be from a deleted file; oracle_list_repos shows when the repo was last indexed.`,
      };
    }
    return {
      ok: false,
      reason: "read_error",
      message: `Could not read ${absolutePath}: ${(err as Error).message}`,
    };
  }

  const lines = content.split("\n");
  const totalLines = lines.length;
  const requestedWindow = Math.max(1, Math.min(req.window ?? DEFAULT_WINDOW, MAX_WINDOW));
  const center = Math.max(1, Math.min(req.line ?? 1, totalLines));
  // Symmetric window around `center`.
  const half = Math.floor(requestedWindow / 2);
  const lineStart = Math.max(1, center - half);
  const lineEnd = Math.min(totalLines, lineStart + requestedWindow - 1);

  return {
    ok: true,
    repo: req.repo,
    path: req.path,
    lineStart,
    lineEnd,
    totalLines,
    text: renderCatN(lines, lineStart, lineEnd),
  };
}

/** Render the requested line range in cat-n style: right-aligned line number, arrow, content. */
function renderCatN(lines: string[], lineStart: number, lineEnd: number): string {
  // Width of the largest line number we'll print, to right-align consistently.
  const width = String(lineEnd).length;
  const out: string[] = [];
  for (let i = lineStart; i <= lineEnd; i++) {
    const num = String(i).padStart(width, " ");
    // Strip a trailing CR so CRLF-encoded files render cleanly in plain text.
    const raw = lines[i - 1] ?? "";
    const content = raw.endsWith("\r") ? raw.slice(0, -1) : raw;
    out.push(`${num}→${content}`);
  }
  return out.join("\n");
}

/** Format an ExpandResult for plain-text MCP/CLI rendering. */
export function formatExpandResult(result: ExpandResult): string {
  if (!result.ok) {
    return `oracle_expand: ${result.message}`;
  }
  // Mirror the oracle_search convention: `path:line_start-line_end (repo)`
  // and append total-lines as a sanity check for the agent.
  const header = `${result.path}:${result.lineStart}-${result.lineEnd} (${result.repo}, ${result.totalLines} lines total)`;
  return `${header}\n\n${result.text}`;
}
