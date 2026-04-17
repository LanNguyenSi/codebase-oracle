import { ChatAnthropic } from "@langchain/anthropic";
import { ChatOpenAI } from "@langchain/openai";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import type { Document } from "@langchain/core/documents";
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

export async function queryCodebase(
  question: string,
  vectorStore: VectorStoreWrapper,
  config: Config,
  options?: { repo?: string; limit?: number },
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
      const { repo, filePath } = doc.metadata as { repo: string; filePath: string };
      return `[${i + 1}] ${filePath} (${repo}):\n\`\`\`\n${doc.pageContent}\n\`\`\``;
    })
    .join("\n\n");

  // Build LLM
  const llm = createLlm(config);

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
      const { filePath } = doc.metadata as { filePath: string };
      return `### ${filePath}\n\`\`\`\n${doc.pageContent.slice(0, 500)}\n\`\`\``;
    })
    .join("\n\n");
}

export function getLlmErrorDetails(err: unknown): string | null {
  if (!err || typeof err !== "object") return null;
  const e = err as {
    status?: number;
    requestID?: string;
    message?: string;
  };
  const parts: string[] = [];
  if (typeof e.status === "number") {
    parts.push(`status ${e.status}`);
  }
  if (typeof e.requestID === "string" && e.requestID.length > 0) {
    parts.push(`request id ${e.requestID}`);
  }
  if (parts.length > 0) return parts.join(", ");
  if (typeof e.message === "string" && e.message.length > 0) return e.message;
  return null;
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

function createOllamaLlm(config: Config) {
  return new ChatOpenAI({
    apiKey: config.ollamaApiKey ?? config.openaiApiKey ?? "ollama",
    modelName: config.llmModel,
    temperature: 0,
    configuration: {
      baseURL: ensureV1BaseUrl(config.ollamaBaseUrl),
    },
  });
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

  if (config.llmProvider === "ollama") {
    return createOllamaLlm(config);
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

export async function searchCodebase(
  query: string,
  vectorStore: VectorStoreWrapper,
  options?: { repo?: string; limit?: number },
): Promise<Document[]> {
  const k = options?.limit ?? 10;
  const filter = options?.repo ? { repo: options.repo } : undefined;
  return vectorStore.similaritySearch(query, k, filter);
}
