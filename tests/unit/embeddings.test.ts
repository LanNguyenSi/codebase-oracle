import { describe, it, expect, afterEach } from "vitest";
import { Embeddings } from "@langchain/core/embeddings";
import { OpenAIEmbeddings } from "@langchain/openai";
import { createEmbeddings } from "../../src/store/embeddings.js";
import type { Config } from "../../src/config.js";

function baseConfig(overrides: Partial<Config> = {}): Config {
  return {
    scanRoot: "/tmp/test",
    dataDir: "/tmp/oracle-test-data",
    embeddingProvider: "openai",
    llmProvider: "auto",
    ollamaBaseUrl: "http://localhost:11434/v1",
    embeddingModel: "text-embedding-3-small",
    embeddingQueryPrefix: "",
    llmModel: "claude-sonnet-4-6",
    maxFileSizeBytes: 500_000,
    maxTextFileSizeBytes: 2_000_000,
    llmTimeoutMs: 120_000,
    ...overrides,
  };
}

describe("createEmbeddings", () => {
  const originalNodeEnv = process.env.NODE_ENV;

  afterEach(() => {
    if (originalNodeEnv === undefined) {
      delete process.env.NODE_ENV;
    } else {
      process.env.NODE_ENV = originalNodeEnv;
    }
  });

  it("provider=stub in non-production returns a StubEmbeddings instance", () => {
    process.env.NODE_ENV = "test";
    const config = baseConfig({ embeddingProvider: "stub" });
    const embeddings = createEmbeddings(config);
    expect(embeddings).toBeInstanceOf(Embeddings);
    expect(embeddings.constructor.name).toBe("StubEmbeddings");
  });

  it("provider=stub in NODE_ENV=production throws the security guard error", () => {
    process.env.NODE_ENV = "production";
    const config = baseConfig({ embeddingProvider: "stub" });
    expect(() => createEmbeddings(config)).toThrow(
      /for tests only.*refused when NODE_ENV=production/i,
    );
  });

  it("provider=openai without openaiApiKey throws OPENAI_API_KEY error", () => {
    process.env.NODE_ENV = "test";
    const config = baseConfig({ embeddingProvider: "openai", openaiApiKey: undefined });
    expect(() => createEmbeddings(config)).toThrow(/OPENAI_API_KEY is required/);
  });

  it("provider=openai with a key returns an OpenAIEmbeddings instance", () => {
    process.env.NODE_ENV = "test";
    const config = baseConfig({ embeddingProvider: "openai", openaiApiKey: "sk-test" });
    const embeddings = createEmbeddings(config);
    expect(embeddings).toBeInstanceOf(OpenAIEmbeddings);
  });

  it("provider=openai carries the configured embeddingModel", () => {
    process.env.NODE_ENV = "test";
    const config = baseConfig({
      embeddingProvider: "openai",
      openaiApiKey: "sk-test",
      embeddingModel: "text-embedding-3-large",
    });
    const embeddings = createEmbeddings(config) as OpenAIEmbeddings & {
      model?: string;
      modelName?: string;
    };
    const model = embeddings.model ?? embeddings.modelName;
    expect(model).toBe("text-embedding-3-large");
  });

  it("provider=ollama returns OpenAIEmbeddings pointing at the ollama base URL with /v1 appended", () => {
    process.env.NODE_ENV = "test";
    const config = baseConfig({
      embeddingProvider: "ollama",
      ollamaBaseUrl: "http://localhost:11434",
      embeddingModel: "nomic-embed-text",
    });
    const embeddings = createEmbeddings(config);
    expect(embeddings).toBeInstanceOf(OpenAIEmbeddings);
    const cfg = embeddings as unknown as {
      clientConfig?: { baseURL?: string };
      configuration?: { baseURL?: string };
    };
    const baseURL = cfg.clientConfig?.baseURL ?? cfg.configuration?.baseURL;
    expect(baseURL).toBe("http://localhost:11434/v1");
  });
});

// The stub provider hashes the input text into a deterministic vector, so
// prefix behaviour is directly observable: prefixed embedQuery(t) must
// equal stub embedQuery(prefix + t), and embedDocuments must be exactly
// the unprefixed embeddings. This must hold for every provider; stub is
// what proves it without a network.
describe("createEmbeddings embeddingQueryPrefix (query-only prefixing, stub)", () => {
  const originalNodeEnv = process.env.NODE_ENV;
  const PREFIX = "Instruct: Given a question about a codebase, retrieve the code that answers it Query: ";

  afterEach(() => {
    if (originalNodeEnv === undefined) {
      delete process.env.NODE_ENV;
    } else {
      process.env.NODE_ENV = originalNodeEnv;
    }
  });

  function stub(prefix: string | undefined): Embeddings {
    process.env.NODE_ENV = "test";
    return createEmbeddings(
      baseConfig({ embeddingProvider: "stub", embeddingQueryPrefix: prefix }),
    );
  }

  it("non-empty prefix: embedQuery(t) returns the embedding of prefix + t", async () => {
    const prefixed = stub(PREFIX);
    const plain = stub("");
    const query = "how does the retry loop work";
    expect(await prefixed.embedQuery(query)).toEqual(
      await plain.embedQuery(PREFIX + query),
    );
    expect(await prefixed.embedQuery(query)).not.toEqual(
      await plain.embedQuery(query),
    );
  });

  it("non-empty prefix: embedDocuments stays exactly the unprefixed embeddings", async () => {
    const prefixed = stub(PREFIX);
    const plain = stub("");
    const docs = ["alpha source file", "beta source file"];
    expect(await prefixed.embedDocuments(docs)).toEqual(
      await plain.embedDocuments(docs),
    );
    expect(await prefixed.embedDocuments(docs)).not.toEqual(
      await plain.embedDocuments(docs.map((d) => PREFIX + d)),
    );
  });

  it("empty prefix: embedQuery behaves exactly as today and the provider client is returned un-wrapped", async () => {
    const config = baseConfig({ embeddingProvider: "stub" });
    const embeddings = createEmbeddings(config);
    expect(embeddings.constructor.name).toBe("StubEmbeddings");
    const query = "unchanged query path";
    expect(await embeddings.embedQuery(query)).toEqual(
      await stub(undefined).embedQuery(query),
    );
  });

  it("unset prefix behaves exactly like the empty prefix", async () => {
    const embeddings = stub(undefined);
    expect(embeddings.constructor.name).toBe("StubEmbeddings");
    const query = "same query path";
    expect(await embeddings.embedQuery(query)).toEqual(
      await stub("").embedQuery(query),
    );
  });

  it("the wrapper still satisfies the Embeddings contract for non-stub providers", () => {
    process.env.NODE_ENV = "test";
    const embeddings = createEmbeddings(
      baseConfig({
        embeddingProvider: "openai",
        openaiApiKey: "sk-test",
        embeddingQueryPrefix: PREFIX,
      }),
    );
    expect(embeddings).toBeInstanceOf(Embeddings);
    expect(embeddings).not.toBeInstanceOf(OpenAIEmbeddings);
  });
});
