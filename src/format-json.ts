import type { Document } from "@langchain/core/documents";
import type { ExpandResult } from "./expand.js";
import type { QueryResult } from "./retrieval/chain.js";
import type { IndexedRepo } from "./store/vector-store.js";

function stringArray(value: unknown): string[] | null {
  return Array.isArray(value) && value.every((item) => typeof item === "string")
    ? value
    : null;
}

export function formatSearchJson(
  query: string,
  repo: string | undefined,
  limit: number,
  docs: Document[],
): string {
  return JSON.stringify({
    ok: true,
    query,
    repo: repo ?? null,
    limit,
    results: docs.map((doc) => ({
      repo: typeof doc.metadata.repo === "string" ? doc.metadata.repo : null,
      filePath:
        typeof doc.metadata.filePath === "string" ? doc.metadata.filePath : null,
      lineStart: typeof doc.metadata.lineStart === "number" ? doc.metadata.lineStart : null,
      lineEnd: typeof doc.metadata.lineEnd === "number" ? doc.metadata.lineEnd : null,
      fmType:
        typeof doc.metadata.fmType === "string" ? doc.metadata.fmType : null,
      fmTags: stringArray(doc.metadata.fmTags),
      fmSources: stringArray(doc.metadata.fmSources),
      expandedFrom:
        typeof doc.metadata.expandedFrom === "string"
          ? doc.metadata.expandedFrom
          : null,
      text: doc.pageContent,
    })),
  });
}

export function formatReposJson(repos: IndexedRepo[]): string {
  return JSON.stringify({ ok: true, repos });
}

export function formatExpandJson(result: ExpandResult): string {
  return JSON.stringify(result);
}

export function formatQueryJson(question: string, result: QueryResult): string {
  return JSON.stringify({
    ok: true,
    question,
    answer: result.answer,
    sources: result.sources.map(({ filePath, repo }) => ({ filePath, repo })),
    pointers: result.pointers,
    // Additive: only present (and true) on the LLM-failure raw-context
    // fallback branch of queryCodebase(). Absent on every other branch, so
    // JSON.stringify omits the key and pre-existing consumers see no change.
    ...(result.degraded ? { degraded: true, degradedReason: result.degradedReason } : {}),
  });
}

export function formatErrorJson(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  return JSON.stringify({ ok: false, error: { message } });
}
