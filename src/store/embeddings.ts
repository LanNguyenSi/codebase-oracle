import { OpenAIEmbeddings } from "@langchain/openai";
import { Embeddings } from "@langchain/core/embeddings";
import { createHash } from "node:crypto";
import type { Config } from "../config.js";

const STUB_DIMENSION = 8;

class StubEmbeddings extends Embeddings {
  constructor() {
    super({});
  }

  async embedDocuments(texts: string[]): Promise<number[][]> {
    return texts.map((t) => this.hashToVec(t));
  }

  async embedQuery(text: string): Promise<number[]> {
    return this.hashToVec(text);
  }

  private hashToVec(text: string): number[] {
    // 32 bytes from sha256 → first STUB_DIMENSION bytes folded to floats in
    // [-1, 1]. Deterministic, cheap, no network. Not for production use.
    const hash = createHash("sha256").update(text).digest();
    const out = new Array<number>(STUB_DIMENSION);
    for (let i = 0; i < STUB_DIMENSION; i++) {
      out[i] = (hash[i] - 128) / 128;
    }
    return out;
  }
}

function ensureV1BaseUrl(baseUrl: string): string {
  const trimmed = baseUrl.replace(/\/+$/, "");
  return trimmed.endsWith("/v1") ? trimmed : `${trimmed}/v1`;
}

export function createEmbeddings(config: Config): Embeddings {
  if (config.embeddingProvider === "stub") {
    // Test-only path. Refuse to produce a stub provider in production so a
    // stray ORACLE_EMBEDDING_PROVIDER=stub export can't quietly populate a
    // real index with hash vectors.
    if (process.env.NODE_ENV === "production") {
      throw new Error(
        "ORACLE_EMBEDDING_PROVIDER=stub is for tests only and is refused when NODE_ENV=production.",
      );
    }
    return new StubEmbeddings();
  }

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
