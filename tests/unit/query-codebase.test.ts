import { describe, it, expect } from "vitest";
import { Document } from "@langchain/core/documents";
import {
  queryCodebase,
  formatRawContextAnswer,
  extractSources,
  createLlm,
} from "../../src/retrieval/chain.js";
import type { Config } from "../../src/config.js";
import type { VectorStoreWrapper } from "../../src/store/vector-store.js";

function baseConfig(overrides: Partial<Config> = {}): Config {
  return {
    scanRoot: "/tmp/test",
    dataDir: "/tmp/oracle-test-data",
    embeddingProvider: "openai",
    llmProvider: "auto",
    ollamaBaseUrl: "http://localhost:11434/v1",
    embeddingModel: "text-embedding-3-small",
    llmModel: "claude-sonnet-4-6",
    vectorStoreType: "directory",
    maxFileSizeBytes: 500_000,
    maxTextFileSizeBytes: 2_000_000,
    ...overrides,
  };
}

function stubStore(docs: Document[]): VectorStoreWrapper {
  return {
    similaritySearch: async (
      _query: string,
      _k?: number,
      _filter?: Record<string, string>,
    ) => docs,
    addDocuments: async () => {},
    listRepos: () => [],
    getFileMetadata: () => null,
    getFirstChunkByFile: () => null,
    close: () => {},
  };
}

function makeDoc(
  filePath: string,
  pageContent: string,
  repo = "test-repo",
): Document {
  return new Document({ pageContent, metadata: { filePath, repo } });
}

describe("queryCodebase", () => {
  describe("empty-docs branch", () => {
    it("returns the no-results message and empty sources when similaritySearch returns []", async () => {
      const store = stubStore([]);
      const config = baseConfig();
      const result = await queryCodebase("what does this do?", store, config);
      expect(result.answer).toMatch(/No relevant code found/);
      expect(result.sources).toEqual([]);
    });

    it("empty sources list is exactly [] (not falsy or undefined)", async () => {
      const store = stubStore([]);
      const result = await queryCodebase("x", store, baseConfig());
      expect(Array.isArray(result.sources)).toBe(true);
      expect(result.sources).toHaveLength(0);
    });

    it("pointers is exactly [] when there are no retrieved docs", async () => {
      const store = stubStore([]);
      const result = await queryCodebase("x", store, baseConfig());
      expect(result.pointers).toEqual([]);
    });
  });

  describe("no-LLM raw branch (auto mode, no API keys → createLlm returns null)", () => {
    it("returns formatRawContextAnswer output when no LLM is available", async () => {
      const docs = [
        makeDoc("src/a.ts", "function hello() {}", "repo-a"),
        makeDoc("src/b.ts", "function world() {}", "repo-a"),
      ];
      const store = stubStore(docs);
      // auto + no anthropicApiKey + no openaiApiKey → createLlm returns null
      const config = baseConfig({
        llmProvider: "auto",
        anthropicApiKey: undefined,
        openaiApiKey: undefined,
      });
      const result = await queryCodebase("what does this do?", store, config);
      expect(result.answer).toBe(formatRawContextAnswer(docs));
    });

    it("extracts sources from docs in the no-LLM path", async () => {
      const docs = [makeDoc("src/x.ts", "const x = 1;", "my-repo")];
      const store = stubStore(docs);
      const config = baseConfig({
        llmProvider: "auto",
        anthropicApiKey: undefined,
        openaiApiKey: undefined,
      });
      const result = await queryCodebase("x?", store, config);
      expect(result.sources).toEqual(extractSources(docs));
      expect(result.sources).toHaveLength(1);
      expect(result.sources[0].repo).toBe("my-repo");
      expect(result.sources[0].filePath).toBe("src/x.ts");
    });

    it("deduplicates sources by filePath in the no-LLM path", async () => {
      const docs = [
        makeDoc("src/shared.ts", "// first chunk", "repo"),
        makeDoc("src/shared.ts", "// second chunk", "repo"),
      ];
      const store = stubStore(docs);
      const config = baseConfig({
        llmProvider: "auto",
        anthropicApiKey: undefined,
        openaiApiKey: undefined,
      });
      const result = await queryCodebase("shared?", store, config);
      // extractSources dedupes by filePath: two docs with the same path → one source
      expect(result.sources).toHaveLength(1);
    });
  });

  describe("pointers (OKF fmSources)", () => {
    const noLlmConfig = baseConfig({
      llmProvider: "auto",
      anthropicApiKey: undefined,
      openaiApiKey: undefined,
    });

    it("is [] when no retrieved chunk carries fmSources", async () => {
      const docs = [makeDoc("src/a.ts", "code", "repo")];
      const store = stubStore(docs);
      const result = await queryCodebase("x?", store, noLlmConfig);
      expect(result.pointers).toEqual([]);
    });

    it("collects the deduped, rank-ordered union of fmSources across retrieved chunks", async () => {
      const docs = [
        new Document({
          pageContent: "a",
          metadata: {
            filePath: "docs/a.md",
            repo: "docs",
            fmSources: ["src2", "src1"],
          },
        }),
        new Document({
          pageContent: "b",
          metadata: {
            filePath: "docs/b.md",
            repo: "docs",
            fmSources: ["src1", "src3"],
          },
        }),
      ];
      const store = stubStore(docs);
      const result = await queryCodebase("x?", store, noLlmConfig);
      expect(result.pointers).toEqual(["src2", "src1", "src3"]);
    });
  });
});

// ── LLM invoke-failure branch (via deps injection seam) ───────────────────────
//
// The `deps.createLlm` param (added in this task) allows tests to inject a
// factory that returns a value chain.invoke() will reject for, exercising the
// catch path without making a real LLM call. The injected factory returns a
// plain async function; LangChain's Runnable.pipe() coerces plain functions
// to RunnableLambda, so prompt.pipe(fn) works and fn's throw propagates to
// the catch block.

describe("queryCodebase — LLM invoke-failure branch (deps seam)", () => {
  it("returns 'LLM request failed' answer with raw context when chain.invoke rejects", async () => {
    const docs = [
      makeDoc("src/a.ts", "const x = 1;", "repo-x"),
      makeDoc("src/b.ts", "const y = 2;", "repo-x"),
    ];
    const store = stubStore(docs);
    // Use a config that would normally produce a real LLM (anthropic key present)
    // but override createLlm so we never hit the network.
    const config = baseConfig({
      llmProvider: "anthropic",
      anthropicApiKey: "sk-ant-test",
    });

    // Inject a createLlm that returns a throwing async function.
    // LangChain's pipe() accepts RunnableFunc (plain async function) as a step,
    // so prompt.pipe(throwingFn) builds a chain whose invoke() rejects.
    const throwingFn = async (_input: unknown): Promise<never> => {
      throw new Error("ECONNREFUSED 127.0.0.1:11434");
    };
    const fakeLlmFactory = (_cfg: Config) =>
      throwingFn as unknown as ReturnType<typeof createLlm>;

    const result = await queryCodebase("what is x?", store, config, undefined, {
      createLlm: fakeLlmFactory as typeof createLlm,
    });

    // Catch block formats: "LLM request failed<details>. Returning raw retrieved context..."
    expect(result.answer).toMatch(/^LLM request failed/);
    expect(result.answer).toContain("Returning raw retrieved context instead");
    // Raw context should include the docs
    expect(result.answer).toContain("src/a.ts");
    // Sources are still extracted from docs
    expect(result.sources).toEqual(extractSources(docs));
    expect(result.sources).toHaveLength(2);
    // No fm metadata on these docs: pointers stays empty.
    expect(result.pointers).toEqual([]);
  });

  it("still propagates pointers from fmSources when the LLM call fails", async () => {
    const docs = [
      new Document({
        pageContent: "a",
        metadata: { filePath: "docs/a.md", repo: "docs", fmSources: ["src1"] },
      }),
    ];
    const store = stubStore(docs);
    const config = baseConfig({
      llmProvider: "anthropic",
      anthropicApiKey: "sk-ant-test",
    });
    const throwingFn = async (_input: unknown): Promise<never> => {
      throw new Error("boom");
    };
    const fakeLlmFactory = (_cfg: Config) =>
      throwingFn as unknown as ReturnType<typeof createLlm>;

    const result = await queryCodebase("what is x?", store, config, undefined, {
      createLlm: fakeLlmFactory as typeof createLlm,
    });

    expect(result.pointers).toEqual(["src1"]);
  });
});

// ── LLM invoke-SUCCESS branch (via the same deps injection seam) ─────────────
//
// The empty-docs, no-LLM-raw, and invoke-failure branches all had pointer
// coverage, but a mutation that returned `pointers: []` specifically in the
// final (LLM answered successfully) return statement would have escaped the
// suite. This exercises that exact branch: the injected function RESOLVES
// (instead of throwing), so chain.invoke() succeeds and queryCodebase reaches
// its final `return { answer, sources, pointers }`.

describe("queryCodebase, LLM invoke-SUCCESS branch (deps seam)", () => {
  it("propagates pointers from fmSources alongside a real LLM answer", async () => {
    const docs = [
      new Document({
        pageContent: "a",
        metadata: {
          filePath: "docs/a.md",
          repo: "docs",
          fmSources: ["src2", "src1"],
        },
      }),
      new Document({
        pageContent: "b",
        metadata: { filePath: "docs/b.md", repo: "docs", fmSources: ["src1"] },
      }),
    ];
    const store = stubStore(docs);
    const config = baseConfig({
      llmProvider: "anthropic",
      anthropicApiKey: "sk-ant-test",
    });

    // A plain async function resolving to a string: LangChain's pipe()
    // coerces it to a RunnableLambda, and the subsequent StringOutputParser
    // step passes a string chunk straight through, so chain.invoke()
    // resolves to exactly this string.
    const resolvingFn = async (_input: unknown): Promise<string> =>
      "the LLM answer";
    const fakeLlmFactory = (_cfg: Config) =>
      resolvingFn as unknown as ReturnType<typeof createLlm>;

    const result = await queryCodebase("what is x?", store, config, undefined, {
      createLlm: fakeLlmFactory as typeof createLlm,
    });

    expect(result.answer).toBe("the LLM answer");
    expect(result.pointers).toEqual(["src2", "src1"]);
  });
});
