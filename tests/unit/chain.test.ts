import { afterEach, beforeEach, describe, it, expect, vi } from "vitest";
import { Document } from "@langchain/core/documents";
import {
  createLlm,
  extractSources,
  formatChunkLocation,
  formatRawContextAnswer,
  getLlmErrorDetails,
  resetOllamaDeprecationWarning,
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
    llmModel: "claude-sonnet-4-6",
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
      llmModel: "claude-sonnet-4-6",
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

  // The next block exercises provider=openai-compatible and the legacy
  // ollama alias. They share state (the one-shot deprecation warning),
  // so reset between tests.
  beforeEach(() => {
    resetOllamaDeprecationWarning();
  });
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("provider=openai-compatible requires llmBaseUrl", () => {
    const config = baseConfig({
      llmProvider: "openai-compatible",
      llmApiKey: "gsk-test",
      llmModel: "llama-3.3-70b-versatile",
      // Override the ollamaBaseUrl default to undefined to exercise the
      // missing-baseUrl path. (Zod normally fills the default; we strip it.)
    });
    (config as Partial<Config>).ollamaBaseUrl = undefined;
    expect(() => createLlm(config)).toThrow(/ORACLE_LLM_BASE_URL/);
  });

  it("provider=openai-compatible builds a ChatOpenAI with llmBaseUrl + llmApiKey", () => {
    const config = baseConfig({
      llmProvider: "openai-compatible",
      llmBaseUrl: "https://api.groq.com/openai/v1",
      llmApiKey: "gsk-test",
      llmModel: "llama-3.3-70b-versatile",
    });
    const llm = createLlm(config) as
      | { constructor: { name: string }; openAIApiKey?: string; apiKey?: string }
      | null;
    expect(llm?.constructor.name).toBe("ChatOpenAI");
    // The SDK exposes the resolved key on either openAIApiKey or apiKey
    // depending on version; either way the value must not be empty.
    const resolvedKey = llm?.openAIApiKey ?? llm?.apiKey;
    expect(typeof resolvedKey === "string" ? resolvedKey : undefined).toBe("gsk-test");
  });

  it("provider=openai-compatible does NOT consume openaiApiKey from the embedding lane", () => {
    const config = baseConfig({
      llmProvider: "openai-compatible",
      llmBaseUrl: "https://api.example.com/v1",
      openaiApiKey: "sk-embedding-only",
      // No llmApiKey, no ollamaApiKey
    });
    const llm = createLlm(config) as
      | { openAIApiKey?: string; apiKey?: string }
      | null;
    const resolvedKey = llm?.openAIApiKey ?? llm?.apiKey;
    // Should be empty string (fallback), NOT sk-embedding-only.
    expect(resolvedKey).not.toBe("sk-embedding-only");
  });

  it("legacy provider=ollama still picks up ollamaApiKey + ollamaBaseUrl", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const config = baseConfig({
      llmProvider: "ollama",
      ollamaApiKey: "legacy-key",
      ollamaBaseUrl: "http://localhost:11434/v1",
      llmModel: "llama3.1",
    });
    const llm = createLlm(config) as
      | { openAIApiKey?: string; apiKey?: string }
      | null;
    const resolvedKey = llm?.openAIApiKey ?? llm?.apiKey;
    expect(resolvedKey).toBe("legacy-key");
    expect(warn).toHaveBeenCalledTimes(1);
    expect(warn.mock.calls[0]![0]).toMatch(/ORACLE_LLM_PROVIDER=ollama is deprecated/);
  });

  it("new llmApiKey takes precedence over legacy ollamaApiKey", () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    const config = baseConfig({
      llmProvider: "ollama",
      llmApiKey: "new-key",
      ollamaApiKey: "legacy-key",
      ollamaBaseUrl: "http://localhost:11434/v1",
    });
    const llm = createLlm(config) as
      | { openAIApiKey?: string; apiKey?: string }
      | null;
    const resolvedKey = llm?.openAIApiKey ?? llm?.apiKey;
    expect(resolvedKey).toBe("new-key");
  });

  it("deprecation warning is emitted at most once across calls", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const config = baseConfig({
      llmProvider: "ollama",
      ollamaApiKey: "k",
      ollamaBaseUrl: "http://localhost:11434/v1",
    });
    createLlm(config);
    createLlm(config);
    createLlm(config);
    expect(warn).toHaveBeenCalledTimes(1);
  });
});

describe("getLlmErrorDetails", () => {
  it("formats status + request id + message together", () => {
    const err = { status: 500, requestID: "req_abc", message: "boom" };
    expect(getLlmErrorDetails(err)).toBe("status 500, request id req_abc, boom");
  });

  it("returns just status when no request id and no message", () => {
    expect(getLlmErrorDetails({ status: 429 })).toBe("status 429");
  });

  it("returns just request id when no status", () => {
    expect(getLlmErrorDetails({ requestID: "req_xyz" })).toBe("request id req_xyz");
  });

  it("includes the message when no status or request id", () => {
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

  it("returns null when message is empty and status is absent", () => {
    expect(getLlmErrorDetails({ message: "" })).toBeNull();
  });

  it("includes the SDK message even when status + requestID are also set", () => {
    // Previously this dropped the actionable message ('401 unauthorized')
    // behind 'status 401, request id ...'. The fix appends it instead.
    const err = {
      status: 401,
      requestID: "req_auth",
      message: "401 unauthorized",
    };
    expect(getLlmErrorDetails(err)).toBe("status 401, request id req_auth, 401 unauthorized");
  });

  it("trims the message to its first non-empty line", () => {
    const err = {
      status: 401,
      message: "401 unauthorized\n\nTroubleshooting URL: https://example.com/auth",
    };
    expect(getLlmErrorDetails(err)).toBe("status 401, 401 unauthorized");
  });

  it("caps very long single-line messages", () => {
    const longMessage = "model not found: " + "x".repeat(500);
    const detail = getLlmErrorDetails({ status: 404, message: longMessage });
    expect(detail).not.toBeNull();
    // Status prefix + the cap (240 incl. ellipsis).
    expect(detail!.length).toBeLessThan(longMessage.length);
    expect(detail!.endsWith("…")).toBe(true);
  });

  it("surfaces ECONNREFUSED via error.code when no HTTP status is set", () => {
    const err = {
      code: "ECONNREFUSED",
      message: "connect ECONNREFUSED 127.0.0.1:11434",
    };
    expect(getLlmErrorDetails(err)).toBe(
      "ECONNREFUSED, connect ECONNREFUSED 127.0.0.1:11434",
    );
  });

  it("falls back to cause.code when error.code is missing", () => {
    const err = {
      message: "fetch failed",
      cause: { code: "ENOTFOUND", message: "getaddrinfo ENOTFOUND ollama.local" },
    };
    expect(getLlmErrorDetails(err)).toBe("ENOTFOUND, fetch failed");
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

  it("renders chunk locations with line numbers when present", () => {
    const docs = [
      new Document({
        pageContent: "fn",
        metadata: { filePath: "r/a.ts", repo: "r", lineStart: 12, lineEnd: 27 },
      }),
    ];
    const out = formatRawContextAnswer(docs);
    expect(out).toContain("### r/a.ts:12-27");
  });
});

describe("formatChunkLocation", () => {
  it("renders path:start-end when both line numbers are present", () => {
    expect(formatChunkLocation({ filePath: "r/a.ts", lineStart: 1, lineEnd: 30 }))
      .toBe("r/a.ts:1-30");
  });

  it("renders path:line when start equals end", () => {
    expect(formatChunkLocation({ filePath: "r/a.ts", lineStart: 5, lineEnd: 5 }))
      .toBe("r/a.ts:5");
  });

  it("falls back to bare filePath when line numbers are missing", () => {
    expect(formatChunkLocation({ filePath: "r/a.ts" })).toBe("r/a.ts");
  });

  it("falls back when only one of the two line numbers is present", () => {
    expect(formatChunkLocation({ filePath: "r/a.ts", lineStart: 5 })).toBe("r/a.ts");
    expect(formatChunkLocation({ filePath: "r/a.ts", lineEnd: 9 })).toBe("r/a.ts");
  });
});
