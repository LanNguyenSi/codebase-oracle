import { describe, it, expect } from "vitest";
import { Document } from "@langchain/core/documents";
import {
  createLlm,
  extractSources,
  formatRawContextAnswer,
  getLlmErrorDetails,
} from "../../src/retrieval/chain.js";
import type { Config } from "../../src/config.js";

function baseConfig(overrides: Partial<Config> = {}): Config {
  return {
    scanRoot: "/tmp/test",
    dataDir: "/tmp/oracle-test-data",
    embeddingProvider: "openai",
    llmProvider: "auto",
    ollamaBaseUrl: "http://localhost:11434/v1",
    embeddingModel: "text-embedding-3-small",
    llmModel: "claude-sonnet-4-20250514",
    vectorStoreType: "directory",
    ...overrides,
  };
}

describe("createLlm", () => {
  it("throws when provider=anthropic but no ANTHROPIC_API_KEY is set", () => {
    const config = baseConfig({ llmProvider: "anthropic" });
    expect(() => createLlm(config)).toThrow(/ANTHROPIC_API_KEY/);
  });

  it("throws when provider=openai but no OPENAI_API_KEY is set", () => {
    const config = baseConfig({ llmProvider: "openai" });
    expect(() => createLlm(config)).toThrow(/OPENAI_API_KEY/);
  });

  it("returns an OpenAI-flavoured client for provider=ollama without requiring real keys", () => {
    const config = baseConfig({ llmProvider: "ollama", llmModel: "llama3.1" });
    const llm = createLlm(config);
    expect(llm).not.toBeNull();
    expect(llm?.constructor.name).toBe("ChatOpenAI");
  });

  it("auto mode prefers Anthropic when ANTHROPIC_API_KEY is set", () => {
    const config = baseConfig({
      llmProvider: "auto",
      anthropicApiKey: "sk-ant-test",
      openaiApiKey: "sk-test",
    });
    const llm = createLlm(config);
    expect(llm?.constructor.name).toBe("ChatAnthropic");
  });

  it("auto mode falls back to OpenAI with gpt-4o-mini when only OPENAI_API_KEY is set", () => {
    const config = baseConfig({
      llmProvider: "auto",
      openaiApiKey: "sk-test",
      llmModel: "claude-sonnet-4-20250514",
    });
    const llm = createLlm(config) as { constructor: { name: string }; model?: string; modelName?: string } | null;
    expect(llm?.constructor.name).toBe("ChatOpenAI");
    const resolvedModel = llm?.model ?? llm?.modelName;
    expect(resolvedModel).toBe("gpt-4o-mini");
  });

  it("auto mode returns null when neither Anthropic nor OpenAI keys are available", () => {
    const config = baseConfig({ llmProvider: "auto" });
    expect(createLlm(config)).toBeNull();
  });

  it("provider=anthropic with a key returns a ChatAnthropic instance", () => {
    const config = baseConfig({
      llmProvider: "anthropic",
      anthropicApiKey: "sk-ant-test",
    });
    const llm = createLlm(config);
    expect(llm?.constructor.name).toBe("ChatAnthropic");
  });
});

describe("getLlmErrorDetails", () => {
  it("formats status + request id together", () => {
    const err = { status: 500, requestID: "req_abc", message: "boom" };
    expect(getLlmErrorDetails(err)).toBe("status 500, request id req_abc");
  });

  it("returns just status when no request id", () => {
    expect(getLlmErrorDetails({ status: 429 })).toBe("status 429");
  });

  it("returns just request id when no status", () => {
    expect(getLlmErrorDetails({ requestID: "req_xyz" })).toBe("request id req_xyz");
  });

  it("falls back to the message when no status or request id", () => {
    expect(getLlmErrorDetails({ message: "connection reset" })).toBe("connection reset");
  });

  it("ignores empty-string request id", () => {
    expect(getLlmErrorDetails({ status: 502, requestID: "" })).toBe("status 502");
  });

  it("returns null for null, strings, numbers, and empty objects", () => {
    expect(getLlmErrorDetails(null)).toBeNull();
    expect(getLlmErrorDetails("plain string")).toBeNull();
    expect(getLlmErrorDetails(42)).toBeNull();
    expect(getLlmErrorDetails({})).toBeNull();
  });

  it("returns null when message is empty", () => {
    expect(getLlmErrorDetails({ message: "" })).toBeNull();
  });
});

describe("extractSources", () => {
  it("dedupes by filePath across documents", () => {
    const docs = [
      new Document({ pageContent: "a", metadata: { repo: "r1", filePath: "r1/x.ts" } }),
      new Document({ pageContent: "b", metadata: { repo: "r1", filePath: "r1/x.ts" } }),
      new Document({ pageContent: "c", metadata: { repo: "r2", filePath: "r2/y.ts" } }),
    ];
    const sources = extractSources(docs);
    expect(sources).toHaveLength(2);
    expect(sources[0].filePath).toBe("r1/x.ts");
    expect(sources[0].snippet).toBe("a");
    expect(sources[1].filePath).toBe("r2/y.ts");
  });

  it("returns [] for an empty input array", () => {
    expect(extractSources([])).toEqual([]);
  });

  it("does not crash when metadata is missing filePath", () => {
    const docs = [
      new Document({ pageContent: "x", metadata: {} }),
      new Document({ pageContent: "y", metadata: {} }),
    ];
    expect(() => extractSources(docs)).not.toThrow();
    const sources = extractSources(docs);
    // Dedup key is undefined — first entry kept, rest dropped.
    expect(sources).toHaveLength(1);
  });

  it("truncates the snippet to 200 characters", () => {
    const longContent = "a".repeat(500);
    const sources = extractSources([
      new Document({ pageContent: longContent, metadata: { repo: "r", filePath: "r/x.ts" } }),
    ]);
    expect(sources[0].snippet).toHaveLength(200);
  });
});

describe("formatRawContextAnswer", () => {
  it("emits one markdown section per document with the file path as heading", () => {
    const docs = [
      new Document({ pageContent: "function a() {}", metadata: { filePath: "r/a.ts", repo: "r" } }),
      new Document({ pageContent: "function b() {}", metadata: { filePath: "r/b.ts", repo: "r" } }),
    ];
    const out = formatRawContextAnswer(docs);
    expect(out).toContain("### r/a.ts");
    expect(out).toContain("### r/b.ts");
    expect(out).toContain("function a() {}");
    expect(out).toContain("function b() {}");
    // Sections separated by blank line (\n\n join) and wrapped in ``` fences.
    expect(out.match(/```/g)?.length).toBe(4);
  });

  it("truncates the snippet body at 500 characters", () => {
    const longContent = "x".repeat(900);
    const docs = [
      new Document({ pageContent: longContent, metadata: { filePath: "r/long.ts", repo: "r" } }),
    ];
    const out = formatRawContextAnswer(docs);
    // Count the x's between the code fences.
    const fenceMatch = out.match(/```\n(x+)\n```/);
    expect(fenceMatch).not.toBeNull();
    expect(fenceMatch![1]).toHaveLength(500);
  });

  it("returns empty string for empty input", () => {
    expect(formatRawContextAnswer([])).toBe("");
  });
});
