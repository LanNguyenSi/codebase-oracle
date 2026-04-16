import { OpenAIEmbeddings } from "@langchain/openai";
import type { Embeddings } from "@langchain/core/embeddings";
import type { Config } from "../config.js";

function ensureV1BaseUrl(baseUrl: string): string {
  const trimmed = baseUrl.replace(/\/+$/, "");
  return trimmed.endsWith("/v1") ? trimmed : `${trimmed}/v1`;
}

export function createEmbeddings(config: Config): Embeddings {
  if (config.embeddingProvider === "ollama") {
    return new OpenAIEmbeddings({
      apiKey: config.ollamaApiKey ?? config.openaiApiKey ?? "ollama",
      modelName: config.embeddingModel,
      stripNewLines: true,
      configuration: {
        baseURL: ensureV1BaseUrl(config.ollamaBaseUrl),
      },
    });
  }

  if (!config.openaiApiKey) {
    throw new Error(
      "OPENAI_API_KEY is required for embeddings when ORACLE_EMBEDDING_PROVIDER=openai.",
    );
  }

  return new OpenAIEmbeddings({
    openAIApiKey: config.openaiApiKey,
    modelName: config.embeddingModel,
    stripNewLines: true,
    configuration: config.openaiBaseUrl
      ? { baseURL: config.openaiBaseUrl }
      : undefined,
  });
}
