import { ChatAnthropic } from "@langchain/anthropic";
import { ChatOpenAI } from "@langchain/openai";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { Document } from "@langchain/core/documents";
import picomatch from "picomatch";
import type { Config } from "../config.js";
import type { VectorStoreWrapper } from "../store/vector-store.js";

const SYSTEM_PROMPT = `You are Codebase Oracle, an expert on the indexed multi-repo codebase.
You answer questions about the codebase using the retrieved source code and documentation chunks below.

Rules:
- Base your answers ONLY on the provided context chunks. If the context doesn't contain enough information, say so.
- Always cite sources: include the file path for every claim (e.g. "In \`agent-tasks/backend/src/routes/tasks.ts\`...").
- When showing code, reference the repo and file path.
- Be concise and technical. The user is a developer.
- If asked about cross-repo relationships, explain how the pieces connect.

Context chunks:
{context}`;

const USER_PROMPT = `{question}`;
const OPENAI_AUTO_FALLBACK_MODEL = "gpt-4o-mini";

export interface QueryResult {
  answer: string;
  sources: Array<{ repo: string; filePath: string; snippet: string }>;
  // Deduped, rank-ordered fmSources paths collected across every retrieved
  // chunk that contributed to the answer context (uncapped; the 10-item cap
  // and truncation note are applied at render time by
  // formatPointersSection()). Empty when no retrieved chunk carries
  // fmSources metadata.
  pointers: string[];
}

// Format a chunk's path with line numbers when the splitter recorded them.
// Older chunks indexed before the line-number rollout fall back to the bare
// filePath.
export function formatChunkLocation(metadata: Record<string, unknown>): string {
  const filePath =
    typeof metadata.filePath === "string" ? metadata.filePath : "";
  const lineStart =
    typeof metadata.lineStart === "number" ? metadata.lineStart : null;
  const lineEnd =
    typeof metadata.lineEnd === "number" ? metadata.lineEnd : null;
  if (lineStart !== null && lineEnd !== null) {
    return lineStart === lineEnd
      ? `${filePath}:${lineStart}`
      : `${filePath}:${lineStart}-${lineEnd}`;
  }
  return filePath;
}

// Frontmatter text is user-controlled (it lives in the indexed repos'
// markdown files), so a hostile fmType/fmSources value could otherwise
// inject newlines or other control characters into structural, single-line
// search-result output. Collapse any run of ASCII control characters
// (\r, \n, tabs, etc.) to a single space before interpolating.
function sanitizeForDisplay(value: string): string {
  // eslint-disable-next-line no-control-regex
  return value.replace(/[\x00-\x1F\x7F]+/g, " ");
}

// Renders a chunk's fmType (from OKF frontmatter metadata) as a bracketed
// tag for search-result headers, e.g. "[module]". Empty string when fmType
// is absent or not a non-empty string, so callers can splice it in without
// an extra presence check.
export function formatChunkTypeTag(metadata: Record<string, unknown>): string {
  return typeof metadata.fmType === "string" && metadata.fmType.length > 0
    ? `[${sanitizeForDisplay(metadata.fmType)}]`
    : "";
}

// Renders the transient `expandedFrom` marker as a bracketed header tag showing
// the basename of the parent doc that vouched for this file, e.g.
// "[expanded from chain.ts]". The marker is set only on Documents injected by
// sources-expansion (see searchCodebase) and is never persisted. Returns "" when
// the marker is absent, so callers can splice it in alongside the [type] tag
// without an extra presence check.
export function formatChunkExpandedTag(
  metadata: Record<string, unknown>,
): string {
  const expandedFrom = metadata.expandedFrom;
  if (typeof expandedFrom !== "string" || expandedFrom.length === 0) return "";
  const basename = expandedFrom.split("/").pop() || expandedFrom;
  return `[expanded from ${sanitizeForDisplay(basename)}]`;
}

// Renders a chunk's fmSources (from OKF frontmatter metadata) as a single
// "sources: a, b" line for search-result display. Returns null when
// fmSources is absent, not an array, or has no valid (non-empty string)
// entries, so callers can skip the line entirely.
export function formatChunkSourcesLine(
  metadata: Record<string, unknown>,
): string | null {
  const fmSources = metadata.fmSources;
  if (!Array.isArray(fmSources)) return null;
  const paths = fmSources
    .filter((s): s is string => typeof s === "string" && s.length > 0)
    .map(sanitizeForDisplay);
  if (paths.length === 0) return null;
  return `sources: ${paths.join(", ")}`;
}

// Shared search-result renderer for the MCP (stdio) and HTTP MCP surfaces,
// which use an identical header/body layout. Chunks without fm metadata
// render byte-identical to the pre-OKF format: "[i] location (repo):\n<body>".
// Chunks with fmType/fmSources add a "[type]" tag to the header and a
// "sources: ..." line before the body, respectively.
export function formatSearchResults(docs: Document[]): string {
  if (docs.length === 0) return "No results found.";
  return docs
    .map((doc, i) => {
      const { repo } = doc.metadata as { repo: string };
      const location = formatChunkLocation(doc.metadata);
      // A sources-expanded row may carry both the [type] tag and the
      // [expanded from ...] marker; render them in that order. Rows with
      // neither stay byte-identical to the pre-OKF header.
      const tags = [
        formatChunkTypeTag(doc.metadata),
        formatChunkExpandedTag(doc.metadata),
      ].filter((t) => t.length > 0);
      const tagSuffix = tags.length > 0 ? ` ${tags.join(" ")}` : "";
      const header = `[${i + 1}] ${location} (${repo})${tagSuffix}:`;
      const sourcesLine = formatChunkSourcesLine(doc.metadata);
      const body = sourcesLine
        ? `${sourcesLine}\n${doc.pageContent}`
        : doc.pageContent;
      return `${header}\n${body}`;
    })
    .join("\n\n---\n\n");
}

// Collects the union of fmSources across every retrieved chunk, in first-
// appearance order (chunk rank, then within-chunk array order), deduped.
// Uncapped: formatPointersSection() applies the 10-item render cap.
export function extractSourcePointers(docs: Document[]): string[] {
  const seen = new Set<string>();
  const pointers: string[] = [];
  for (const doc of docs) {
    const fmSources = (doc.metadata as Record<string, unknown>).fmSources;
    if (!Array.isArray(fmSources)) continue;
    for (const src of fmSources) {
      if (typeof src !== "string" || src.length === 0) continue;
      if (seen.has(src)) continue;
      seen.add(src);
      pointers.push(src);
    }
  }
  return pointers;
}

const POINTERS_SECTION_LABEL = "Pointers (from OKF sources metadata):";
const POINTERS_CAP = 10;

// Renders the mechanically-assembled OKF pointers section appended after an
// oracle_query answer's sources list. Returns "" (omit entirely) when there
// are no pointers, so today's no-fmSources output stays byte-identical.
export function formatPointersSection(pointers: string[]): string {
  if (pointers.length === 0) return "";
  const shown = pointers.slice(0, POINTERS_CAP);
  const lines = shown.map((p) => `- ${p}`);
  if (pointers.length > POINTERS_CAP) {
    lines.push(`... and ${pointers.length - POINTERS_CAP} more`);
  }
  return `\n\n${POINTERS_SECTION_LABEL}\n${lines.join("\n")}`;
}

// Splits a CLI-style comma-separated option value (e.g. `--tags a,b, c`)
// into a trimmed, non-empty string array. Returns undefined for an absent
// or empty-after-trim value, so "no --tags flag" and "--tags ''" both mean
// "no filter" to callers.
export function parseCommaSeparatedList(
  raw: string | undefined,
): string[] | undefined {
  if (!raw) return undefined;
  const parts = raw
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
  return parts.length > 0 ? parts : undefined;
}

export async function queryCodebase(
  question: string,
  vectorStore: VectorStoreWrapper,
  config: Config,
  options?: { repo?: string; limit?: number },
  deps: { createLlm?: typeof createLlm } = {},
): Promise<QueryResult> {
  const k = options?.limit ?? 12;
  const filter = options?.repo ? { repo: options.repo } : undefined;

  // Retrieve relevant chunks
  const docs = await vectorStore.similaritySearch(question, k, filter);

  if (docs.length === 0) {
    return {
      answer:
        "No relevant code found in the index. Try re-indexing or rephrasing your question.",
      sources: [],
      pointers: [],
    };
  }

  // Mechanically assembled, no LLM involved: union of fmSources across every
  // retrieved chunk, deduped, in retrieval-rank order.
  const pointers = extractSourcePointers(docs);

  // Format context
  const context = docs
    .map((doc, i) => {
      const { repo } = doc.metadata as { repo: string };
      const location = formatChunkLocation(doc.metadata);
      return `[${i + 1}] ${location} (${repo}):\n\`\`\`\n${doc.pageContent}\n\`\`\``;
    })
    .join("\n\n");

  // Build LLM
  const llm = (deps.createLlm ?? createLlm)(config);

  if (!llm) {
    // No LLM available — return raw chunks
    return {
      answer: formatRawContextAnswer(docs),
      sources: extractSources(docs),
      pointers,
    };
  }

  // RAG chain
  const prompt = ChatPromptTemplate.fromMessages([
    ["system", SYSTEM_PROMPT],
    ["human", USER_PROMPT],
  ]);

  const chain = prompt.pipe(llm).pipe(new StringOutputParser());
  let answer: string;
  try {
    answer = await chain.invoke({ context, question });
  } catch (err) {
    const details = getLlmErrorDetails(err);
    const detailText = details ? ` (${details})` : "";
    return {
      answer: `LLM request failed${detailText}. Returning raw retrieved context instead.\n\n${formatRawContextAnswer(docs)}`,
      sources: extractSources(docs),
      pointers,
    };
  }

  return {
    answer,
    sources: extractSources(docs),
    pointers,
  };
}

export function formatRawContextAnswer(docs: Document[]): string {
  return docs
    .map((doc) => {
      const location = formatChunkLocation(doc.metadata);
      return `### ${location}\n\`\`\`\n${doc.pageContent.slice(0, 500)}\n\`\`\``;
    })
    .join("\n\n");
}

export function getLlmErrorDetails(err: unknown): string | null {
  if (!err || typeof err !== "object") return null;
  const e = err as {
    status?: number;
    requestID?: string;
    message?: string;
    code?: string;
    cause?: { code?: string; message?: string };
    // undici wraps dual-stack connect failures as AggregateError with the
    // per-address attempts in `errors[]` and `code` undefined at the top.
    errors?: Array<{ code?: string; message?: string }>;
  };
  const parts: string[] = [];
  if (typeof e.status === "number") {
    parts.push(`status ${e.status}`);
  }
  if (typeof e.requestID === "string" && e.requestID.length > 0) {
    parts.push(`request id ${e.requestID}`);
  }
  // node:net / undici connect failures don't carry an HTTP status. Surfacing
  // the underlying code (ECONNREFUSED, ENOTFOUND, ETIMEDOUT, EAI_AGAIN, ...)
  // tells the caller "the endpoint isn't reachable" instead of just "500".
  const aggregateChild = Array.isArray(e.errors)
    ? e.errors.find(
        (child) =>
          child &&
          typeof child === "object" &&
          typeof child.code === "string" &&
          child.code.length > 0,
      )
    : undefined;
  const networkCode =
    typeof e.code === "string" && e.code.length > 0
      ? e.code
      : typeof e.cause?.code === "string" && e.cause.code.length > 0
        ? e.cause.code
        : (aggregateChild?.code ?? null);
  if (networkCode) {
    parts.push(networkCode);
  }
  // The SDK message frequently carries the operator-actionable reason
  // ("401 unauthorized", "model X not found", "context length exceeded").
  // Previously we returned it only when status/requestID were absent, which
  // hid the real cause behind an opaque "status 500". Always include it,
  // trimmed and capped so the wrapper stays single-line-ish.
  // Use `||` so an empty top-level message falls through to cause / aggregate
  // children instead of short-circuiting at the empty string.
  const message = pickShortMessage(
    e.message || e.cause?.message || aggregateChild?.message,
  );
  if (message) {
    parts.push(message);
  }
  return parts.length > 0 ? parts.join(", ") : null;
}

function pickShortMessage(raw: string | undefined): string | null {
  if (typeof raw !== "string") return null;
  // Take the first non-empty line and cap the length — SDK messages
  // sometimes append multi-paragraph troubleshooting URLs that drown the
  // useful first line.
  const firstLine = raw
    .split(/\r?\n/)
    .map((s) => s.trim())
    .find(Boolean);
  if (!firstLine) return null;
  const MAX = 240;
  return firstLine.length > MAX ? firstLine.slice(0, MAX - 1) + "…" : firstLine;
}

function ensureV1BaseUrl(baseUrl: string): string {
  const trimmed = baseUrl.replace(/\/+$/, "");
  return trimmed.endsWith("/v1") ? trimmed : `${trimmed}/v1`;
}

function createAnthropicLlm(config: Config) {
  return new ChatAnthropic({
    anthropicApiKey: config.anthropicApiKey!,
    modelName: config.llmModel,
    temperature: 0,
    maxTokens: 4096,
  });
}

function createOpenAILlm(config: Config, modelName: string) {
  return new ChatOpenAI({
    openAIApiKey: config.openaiApiKey!,
    modelName,
    temperature: 0,
    configuration: config.openaiBaseUrl
      ? { baseURL: config.openaiBaseUrl }
      : undefined,
  });
}

// Both `openai-compatible` (preferred) and `ollama` (legacy alias) route here.
// Resolution prefers the new ORACLE_LLM_* env pair and falls back to the
// legacy ollama-named pair so existing setups keep working without edits.
// We deliberately do NOT fall back to `openaiApiKey` — embedding-key reuse
// for the LLM was the old footgun (PR for task 5e4eb1f3).
function createOpenAICompatibleLlm(config: Config, isLegacyOllama: boolean) {
  if (isLegacyOllama) {
    warnOllamaProviderDeprecated();
  }
  const baseURL = ensureV1BaseUrl(config.llmBaseUrl ?? config.ollamaBaseUrl);
  // For local Ollama the conventional sentinel key is the literal "ollama"
  // (the server ignores it). For openai-compatible we prefer to leave the
  // key blank and let the SDK surface a clear 401 if the endpoint requires
  // auth — the error-surfacing fix from task 7549a1ce now makes that
  // legible.
  const fallbackKey = isLegacyOllama ? "ollama" : "";
  const apiKey = config.llmApiKey ?? config.ollamaApiKey ?? fallbackKey;
  return new ChatOpenAI({
    apiKey,
    modelName: config.llmModel,
    temperature: 0,
    configuration: { baseURL },
  });
}

let ollamaDeprecationWarned = false;
function warnOllamaProviderDeprecated(): void {
  if (ollamaDeprecationWarned) return;
  ollamaDeprecationWarned = true;
  // eslint-disable-next-line no-console
  console.warn(
    "[codebase-oracle] ORACLE_LLM_PROVIDER=ollama is deprecated. " +
      "Use ORACLE_LLM_PROVIDER=openai-compatible with ORACLE_LLM_BASE_URL + " +
      "ORACLE_LLM_API_KEY instead. The legacy ORACLE_OLLAMA_BASE_URL / " +
      "OLLAMA_API_KEY still resolve as fallbacks for now.",
  );
}

// Test helper. Resets the once-only deprecation guard so unit tests can
// assert the warning fires under fresh module state without a full reimport.
export function resetOllamaDeprecationWarning(): void {
  ollamaDeprecationWarned = false;
}

export function createLlm(config: Config) {
  if (config.llmProvider === "anthropic") {
    if (!config.anthropicApiKey) {
      throw new Error(
        "ORACLE_LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY.",
      );
    }
    return createAnthropicLlm(config);
  }

  if (config.llmProvider === "openai") {
    if (!config.openaiApiKey) {
      throw new Error("ORACLE_LLM_PROVIDER=openai requires OPENAI_API_KEY.");
    }
    return createOpenAILlm(config, config.llmModel);
  }

  if (config.llmProvider === "openai-compatible") {
    if (!config.llmBaseUrl && !config.ollamaBaseUrl) {
      throw new Error(
        "ORACLE_LLM_PROVIDER=openai-compatible requires ORACLE_LLM_BASE_URL.",
      );
    }
    return createOpenAICompatibleLlm(config, false);
  }

  if (config.llmProvider === "ollama") {
    return createOpenAICompatibleLlm(config, true);
  }

  if (config.anthropicApiKey) {
    return createAnthropicLlm(config);
  }

  if (config.openaiApiKey) {
    return createOpenAILlm(config, OPENAI_AUTO_FALLBACK_MODEL);
  }

  return null;
}

export function extractSources(docs: Document[]) {
  const seen = new Set<string>();
  return docs
    .map((doc) => {
      const { repo, filePath } = doc.metadata as {
        repo: string;
        filePath: string;
      };
      const key = filePath;
      if (seen.has(key)) return null;
      seen.add(key);
      return { repo, filePath, snippet: doc.pageContent.slice(0, 200) };
    })
    .filter((s): s is NonNullable<typeof s> => s !== null);
}

export interface SearchCodebaseOptions {
  repo?: string;
  limit?: number;
  // Glob filter on the chunk's filePath metadata. Standard picomatch
  // semantics: `*` within a segment, `**` across segments, `?` for a
  // single character, `{a,b}` for alternatives. AND-composed with `repo`.
  pathGlob?: string;
  // fmType (OKF frontmatter metadata) filter: strict equality. Only matches
  // chunks that HAVE fmType set; chunks without frontmatter metadata are
  // excluded when this is set. AND-composed with repo/pathGlob/tags.
  type?: string;
  // fmTags (OKF frontmatter metadata) filter: contains-ALL semantics, every
  // listed tag must appear in the chunk's fmTags. Only matches chunks that
  // HAVE fmTags set; chunks without frontmatter metadata are excluded when
  // this is set. AND-composed with repo/pathGlob/type.
  tags?: string[];
  // Sources-expansion (default true): after the post-filter top-k is computed,
  // each result row that carries fmSources "vouches" for those files — inject a
  // representative chunk per pointed-at file into the result list. Set false to
  // return the raw retrieval result unchanged. When true but no row carries
  // fmSources, the output is byte-identical to false.
  expandSources?: boolean;
}

function matchesTypeFilter(
  metadata: Record<string, unknown>,
  type: string,
): boolean {
  return metadata.fmType === type;
}

function matchesTagsFilter(
  metadata: Record<string, unknown>,
  tags: string[],
): boolean {
  const fmTags = metadata.fmTags;
  if (!Array.isArray(fmTags)) return false;
  return tags.every((tag) => fmTags.includes(tag));
}

// Max representative chunks injected per parent row whose fmSources vouches for
// other files. Keeps a single high-fanout doc from flooding the result list.
const MAX_INJECTIONS_PER_PARENT = 3;

function fileKeyOf(metadata: Record<string, unknown>): string {
  const repo = typeof metadata.repo === "string" ? metadata.repo : "";
  const filePath =
    typeof metadata.filePath === "string" ? metadata.filePath : "";
  return `${repo}::${filePath}`;
}

// Sources-expansion. For each organic result row that carries fmSources, inject
// a representative chunk (the file's first chunk) for each pointed-at file
// immediately after its parent, in fmSources order, at most
// MAX_INJECTIONS_PER_PARENT per parent. Dedup key is (repo, filePath): a file
// already present as an organic hit anywhere, or already injected, is skipped —
// organic wins. Injected Documents carry a transient `expandedFrom` marker
// (the parent's filePath) and are never persisted. The final list is capped at
// `limit`, so injections displace tail organic rows; if the limit is already
// reached by earlier ranks, later injections simply don't fit. When no row
// carries a resolvable fmSources entry the returned list is the organic list
// unchanged (byte-identical to expandSources:false).
function expandSourcesInResults(
  organic: Document[],
  vectorStore: VectorStoreWrapper,
  limit: number,
): Document[] {
  // Seed dedup with every organic hit so organic always wins over an injection.
  const seen = new Set<string>();
  for (const doc of organic) {
    seen.add(fileKeyOf(doc.metadata as Record<string, unknown>));
  }

  const expanded: Document[] = [];
  for (const parent of organic) {
    expanded.push(parent);
    const metadata = parent.metadata as Record<string, unknown>;
    const fmSources = metadata.fmSources;
    if (!Array.isArray(fmSources)) continue;
    const parentRepo = typeof metadata.repo === "string" ? metadata.repo : "";
    const parentFilePath =
      typeof metadata.filePath === "string" ? metadata.filePath : "";
    if (parentRepo.length === 0) continue;

    let injected = 0;
    for (const src of fmSources) {
      if (injected >= MAX_INJECTIONS_PER_PARENT) break;
      if (typeof src !== "string" || src.length === 0) continue;
      // A non-matching entry (directory, glob, typo, absent file) resolves to
      // null and is skipped silently and deterministically.
      const chunk = vectorStore.getFirstChunkByFile(parentRepo, src);
      if (!chunk) continue;
      const key = fileKeyOf(chunk.metadata);
      if (seen.has(key)) continue;
      seen.add(key);
      expanded.push(
        new Document({
          pageContent: chunk.pageContent,
          metadata: { ...chunk.metadata, expandedFrom: parentFilePath },
        }),
      );
      injected++;
    }
  }

  return expanded.slice(0, limit);
}

export async function searchCodebase(
  query: string,
  vectorStore: VectorStoreWrapper,
  options?: SearchCodebaseOptions,
): Promise<Document[]> {
  const k = options?.limit ?? 10;
  const expandSources = options?.expandSources ?? true;
  const filter = options?.repo ? { repo: options.repo } : undefined;

  // Normalize both single-value and array filters the same way: an empty
  // string / empty array means "no filter", not "match only empty values".
  const type =
    options?.type && options.type.length > 0 ? options.type : undefined;
  const tags =
    options?.tags && options.tags.length > 0 ? options.tags : undefined;
  const needsPostFilter =
    Boolean(options?.pathGlob) || Boolean(type) || Boolean(tags);

  // Compute the organic top-k first (raw retrieval, then any post-filter),
  // then optionally apply sources-expansion. Both branches feed the same
  // expansion step so injected chunks appear regardless of the filter path.
  let organic: Document[];
  if (!needsPostFilter) {
    organic = await vectorStore.similaritySearch(query, k, filter);
  } else {
    // Over-fetch so the post-filter still has a chance of returning k
    // results when the combined filters are narrow. Cap to keep the SQLite
    // scan bounded for pathological cases (e.g. limit=50 → 200 fetched).
    // Shared by pathGlob/type/tags: all three AND-compose through this same
    // over-fetched window.
    const FETCH_MULTIPLIER = 4;
    const FETCH_MAX = 200;
    const overFetch = Math.min(k * FETCH_MULTIPLIER, FETCH_MAX);

    const matchesPath = options?.pathGlob
      ? picomatch(options.pathGlob, { dot: true })
      : null;
    const raw = await vectorStore.similaritySearch(query, overFetch, filter);
    const filtered: Document[] = [];
    for (const doc of raw) {
      const metadata = doc.metadata as Record<string, unknown>;
      if (matchesPath) {
        const filePath =
          typeof metadata.filePath === "string" ? metadata.filePath : "";
        if (!matchesPath(filePath)) continue;
      }
      if (type !== undefined && !matchesTypeFilter(metadata, type)) continue;
      if (tags && !matchesTagsFilter(metadata, tags)) continue;
      filtered.push(doc);
      if (filtered.length >= k) break;
    }
    organic = filtered;
  }

  if (!expandSources) return organic;
  return expandSourcesInResults(organic, vectorStore, k);
}
