import { ChatAnthropic } from "@langchain/anthropic";
import { ChatOpenAI } from "@langchain/openai";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import type { Document } from "@langchain/core/documents";
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
}

// Format a chunk's path with line numbers when the splitter recorded them.
// Older chunks indexed before the line-number rollout fall back to the bare
// filePath.
export function formatChunkLocation(metadata: Record<string, unknown>): string {
  const filePath = typeof metadata.filePath === "string" ? metadata.filePath : "";
  const lineStart = typeof metadata.lineStart === "number" ? metadata.lineStart : null;
  const lineEnd = typeof metadata.lineEnd === "number" ? metadata.lineEnd : null;
  if (lineStart !== null && lineEnd !== null) {
    return lineStart === lineEnd
      ? `${filePath}:${lineStart}`
      : `${filePath}:${lineStart}-${lineEnd}`;
  }
  return filePath;
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
      answer: "No relevant code found in the index. Try re-indexing or rephrasing your question.",
      sources: [],
    };
  }

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
    const detailText = details
      ? ` (${details})`
      : "";
    return {
      answer: `LLM request failed${detailText}. Returning raw retrieved context instead.\n\n${formatRawContextAnswer(docs)}`,
      sources: extractSources(docs),
    };
  }

  return {
    answer,
    sources: extractSources(docs),
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
  const networkCode = typeof e.code === "string" && e.code.length > 0
    ? e.code
    : typeof e.cause?.code === "string" && e.cause.code.length > 0
      ? e.cause.code
      : aggregateChild?.code ?? null;
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
  const firstLine = raw.split(/\r?\n/).map((s) => s.trim()).find(Boolean);
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
  const baseURL = ensureV1BaseUrl(
    config.llmBaseUrl ?? config.ollamaBaseUrl,
  );
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
      throw new Error("ORACLE_LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY.");
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
      const { repo, filePath } = doc.metadata as { repo: string; filePath: string };
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
}

export async function searchCodebase(
  query: string,
  vectorStore: VectorStoreWrapper,
  options?: SearchCodebaseOptions,
): Promise<Document[]> {
  const k = options?.limit ?? 10;
  const filter = options?.repo ? { repo: options.repo } : undefined;

  if (!options?.pathGlob) {
    return vectorStore.similaritySearch(query, k, filter);
  }

  // Over-fetch so the post-filter still has a chance of returning k
  // results when the glob is narrow. Cap to keep the SQLite scan
  // bounded for pathological cases (e.g. limit=50 → 200 fetched).
  const FETCH_MULTIPLIER = 4;
  const FETCH_MAX = 200;
  const overFetch = Math.min(k * FETCH_MULTIPLIER, FETCH_MAX);

  const matchesPath = picomatch(options.pathGlob, { dot: true });
  const raw = await vectorStore.similaritySearch(query, overFetch, filter);
  const filtered: Document[] = [];
  for (const doc of raw) {
    const filePath = (doc.metadata as { filePath?: string }).filePath ?? "";
    if (matchesPath(filePath)) {
      filtered.push(doc);
      if (filtered.length >= k) break;
    }
  }
  return filtered;
}
