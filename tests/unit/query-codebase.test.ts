import { describe, it, expect } from "vitest";
import { Document } from "@langchain/core/documents";
import {
  queryCodebase,
  formatRawContextAnswer,
  extractSources,
} from "../../src/retrieval/chain.js";
import type { Config } from "../../src/config.js";
import type { VectorStoreWrapper } from "../../src/store/vector-store.js";

// NOTE: createLlm is defined inside chain.ts (not a separate-module import), so
// ESM-internal references cannot be intercepted via vi.mock on the chain module
// export. The LLM-invoke-failure branch therefore needs a small injection seam
// (e.g. accepting a llmFactory param) before it can be unit-tested. See open
// questions in the task report.

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
    close: () => {},
  };
}

function makeDoc(filePath: string, pageContent: string, repo = "test-repo"): Document {
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
});
