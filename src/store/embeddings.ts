import { OpenAIEmbeddings } from "@langchain/openai";
import { Embeddings } from "@langchain/core/embeddings";
import { createHash } from "node:crypto";
import { DEFAULT_OLLAMA_BASE_URL, type Config } from "../config.js";

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

// Wraps a provider client so the ORACLE_EMBEDDING_QUERY_PREFIX instruction
// lands strictly on the query path: instruction-tuned embedding models
// (e.g. Qwen3-Embedding) expect it in front of the query but not in front
// of the documents it is matched against. embedDocuments therefore
// delegates completely unprefixed; prefixing it here would quietly
// corrupt every future index.
class QueryPrefixedEmbeddings extends Embeddings {
  constructor(
    private readonly base: Embeddings,
    private readonly queryPrefix: string,
  ) {
    super({});
  }

  async embedDocuments(texts: string[]): Promise<number[][]> {
    return this.base.embedDocuments(texts);
  }

  async embedQuery(text: string): Promise<number[]> {
    return this.base.embedQuery(this.queryPrefix + text);
  }
}

// Unset or empty prefix returns the provider client un-wrapped, so the
// default behavior is exactly the pre-prefix behavior. The prefix is
// deliberately not part of the store fingerprint: changing it only affects
// how queries are embedded, so it needs no reindex (but it should stay the
// same for all queries against one index).
export function createEmbeddings(config: Config): Embeddings {
  const embeddings = buildEmbeddings(config);
  if (!config.embeddingQueryPrefix) return embeddings;
  return new QueryPrefixedEmbeddings(embeddings, config.embeddingQueryPrefix);
}

function buildEmbeddings(config: Config): Embeddings {
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
        // `ollamaBaseUrl` has no schema default (see config.ts); this
        // embedding lane keeps the conventional localhost fallback here.
        baseURL: ensureV1BaseUrl(config.ollamaBaseUrl ?? DEFAULT_OLLAMA_BASE_URL),
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
