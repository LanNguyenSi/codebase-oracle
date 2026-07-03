import { afterEach, beforeEach, describe, it, expect, vi } from "vitest";
import { Document } from "@langchain/core/documents";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  createLlm,
  extractSourcePointers,
  extractSources,
  formatChunkLocation,
  formatChunkSourcesLine,
  formatChunkTypeTag,
  formatPointersSection,
  formatRawContextAnswer,
  formatSearchResults,
  getLlmErrorDetails,
  parseCommaSeparatedList,
  resetOllamaDeprecationWarning,
  searchCodebase,
} from "../../src/retrieval/chain.js";
import { createVectorStore } from "../../src/store/vector-store.js";
import type { Config } from "../../src/config.js";
import type { Embeddings } from "@langchain/core/embeddings";

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
    const llm = createLlm(config) as {
      constructor: { name: string };
      model?: string;
      modelName?: string;
    } | null;
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
    const llm = createLlm(config) as {
      constructor: { name: string };
      openAIApiKey?: string;
      apiKey?: string;
    } | null;
    expect(llm?.constructor.name).toBe("ChatOpenAI");
    // The SDK exposes the resolved key on either openAIApiKey or apiKey
    // depending on version; either way the value must not be empty.
    const resolvedKey = llm?.openAIApiKey ?? llm?.apiKey;
    expect(typeof resolvedKey === "string" ? resolvedKey : undefined).toBe(
      "gsk-test",
    );
  });

  it("provider=openai-compatible does NOT consume openaiApiKey from the embedding lane", () => {
    const config = baseConfig({
      llmProvider: "openai-compatible",
      llmBaseUrl: "https://api.example.com/v1",
      openaiApiKey: "sk-embedding-only",
      // No llmApiKey, no ollamaApiKey
    });
    const llm = createLlm(config) as {
      openAIApiKey?: string;
      apiKey?: string;
    } | null;
    const resolvedKey = llm?.openAIApiKey ?? llm?.apiKey;
    // Strict: must be the documented empty-string fallback, not the
    // embedding-lane key. Future regressions that reintroduce ANY other
    // fallback (openaiApiKey, env, etc.) would change this and fail loudly.
    expect(resolvedKey).toBe("");
  });

  it("new llmBaseUrl takes precedence over legacy ollamaBaseUrl", () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    const config = baseConfig({
      llmProvider: "openai-compatible",
      llmBaseUrl: "https://api.groq.com/openai/v1",
      ollamaBaseUrl: "http://localhost:11434/v1",
      llmApiKey: "k",
    });
    const llm = createLlm(config) as {
      clientConfig?: { baseURL?: string };
      configuration?: { baseURL?: string };
    } | null;
    // ChatOpenAI exposes the resolved baseURL via `clientConfig.baseURL`
    // on recent SDK versions and `configuration.baseURL` historically;
    // accept either, but the value must be the new var, not the legacy.
    const baseUrl = llm?.clientConfig?.baseURL ?? llm?.configuration?.baseURL;
    expect(baseUrl).toBe("https://api.groq.com/openai/v1");
  });

  it("legacy provider=ollama still picks up ollamaApiKey + ollamaBaseUrl", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const config = baseConfig({
      llmProvider: "ollama",
      ollamaApiKey: "legacy-key",
      ollamaBaseUrl: "http://localhost:11434/v1",
      llmModel: "llama3.1",
    });
    const llm = createLlm(config) as {
      openAIApiKey?: string;
      apiKey?: string;
    } | null;
    const resolvedKey = llm?.openAIApiKey ?? llm?.apiKey;
    expect(resolvedKey).toBe("legacy-key");
    expect(warn).toHaveBeenCalledTimes(1);
    expect(warn.mock.calls[0]![0]).toMatch(
      /ORACLE_LLM_PROVIDER=ollama is deprecated/,
    );
  });

  it("new llmApiKey takes precedence over legacy ollamaApiKey", () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    const config = baseConfig({
      llmProvider: "ollama",
      llmApiKey: "new-key",
      ollamaApiKey: "legacy-key",
      ollamaBaseUrl: "http://localhost:11434/v1",
    });
    const llm = createLlm(config) as {
      openAIApiKey?: string;
      apiKey?: string;
    } | null;
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
    expect(getLlmErrorDetails(err)).toBe(
      "status 500, request id req_abc, boom",
    );
  });

  it("returns just status when no request id and no message", () => {
    expect(getLlmErrorDetails({ status: 429 })).toBe("status 429");
  });

  it("returns just request id when no status", () => {
    expect(getLlmErrorDetails({ requestID: "req_xyz" })).toBe(
      "request id req_xyz",
    );
  });

  it("includes the message when no status or request id", () => {
    expect(getLlmErrorDetails({ message: "connection reset" })).toBe(
      "connection reset",
    );
  });

  it("ignores empty-string request id", () => {
    expect(getLlmErrorDetails({ status: 502, requestID: "" })).toBe(
      "status 502",
    );
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
    expect(getLlmErrorDetails(err)).toBe(
      "status 401, request id req_auth, 401 unauthorized",
    );
  });

  it("trims the message to its first non-empty line", () => {
    const err = {
      status: 401,
      message:
        "401 unauthorized\n\nTroubleshooting URL: https://example.com/auth",
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
      cause: {
        code: "ENOTFOUND",
        message: "getaddrinfo ENOTFOUND ollama.local",
      },
    };
    expect(getLlmErrorDetails(err)).toBe("ENOTFOUND, fetch failed");
  });

  it("walks AggregateError.errors[] when code is missing at the top and cause levels", () => {
    const err = {
      name: "AggregateError",
      message: "",
      errors: [
        {
          code: "ECONNREFUSED",
          message: "connect ECONNREFUSED 127.0.0.1:11434",
        },
        { code: "ECONNREFUSED", message: "connect ECONNREFUSED ::1:11434" },
      ],
    };
    expect(getLlmErrorDetails(err)).toBe(
      "ECONNREFUSED, connect ECONNREFUSED 127.0.0.1:11434",
    );
  });

  it("prefers top-level code over AggregateError.errors[]", () => {
    const err = {
      code: "ETIMEDOUT",
      message: "connect timeout",
      errors: [{ code: "ECONNREFUSED", message: "fallback child" }],
    };
    expect(getLlmErrorDetails(err)).toBe("ETIMEDOUT, connect timeout");
  });

  it("skips AggregateError children without a code", () => {
    const err = {
      name: "AggregateError",
      message: "",
      errors: [
        { message: "no code here" },
        { code: "ECONNREFUSED", message: "second child has the code" },
      ],
    };
    expect(getLlmErrorDetails(err)).toBe(
      "ECONNREFUSED, second child has the code",
    );
  });
});

describe("extractSources", () => {
  it("dedupes by filePath across documents", () => {
    const docs = [
      new Document({
        pageContent: "a",
        metadata: { repo: "r1", filePath: "r1/x.ts" },
      }),
      new Document({
        pageContent: "b",
        metadata: { repo: "r1", filePath: "r1/x.ts" },
      }),
      new Document({
        pageContent: "c",
        metadata: { repo: "r2", filePath: "r2/y.ts" },
      }),
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
      new Document({
        pageContent: longContent,
        metadata: { repo: "r", filePath: "r/x.ts" },
      }),
    ]);
    expect(sources[0].snippet).toHaveLength(200);
  });
});

describe("formatRawContextAnswer", () => {
  it("emits one markdown section per document with the file path as heading", () => {
    const docs = [
      new Document({
        pageContent: "function a() {}",
        metadata: { filePath: "r/a.ts", repo: "r" },
      }),
      new Document({
        pageContent: "function b() {}",
        metadata: { filePath: "r/b.ts", repo: "r" },
      }),
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
      new Document({
        pageContent: longContent,
        metadata: { filePath: "r/long.ts", repo: "r" },
      }),
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

describe("searchCodebase path-glob filter", () => {
  // Hand-rolled stub store: similaritySearch returns the fixed doc list
  // in the order given, regardless of query. Lets us assert the glob
  // filter independently of the embedder.
  function stubStore(docs: Document[]) {
    return {
      similaritySearch: async (
        _query: string,
        k: number,
        filter?: Record<string, string>,
      ): Promise<Document[]> => {
        const scoped = filter?.repo
          ? docs.filter(
              (d) => (d.metadata as { repo?: string }).repo === filter.repo,
            )
          : docs;
        return scoped.slice(0, k);
      },
    };
  }

  function makeDoc(filePath: string, repo = "demo"): Document {
    return new Document({
      pageContent: `// ${filePath}`,
      metadata: { filePath, repo },
    });
  }

  it("returns unfiltered results when pathGlob is absent", async () => {
    const store = stubStore([
      makeDoc("src/a.ts"),
      makeDoc("docs/b.md"),
      makeDoc(".github/workflows/release.yml"),
    ]);
    const out = await searchCodebase("x", store as never, { limit: 3 });
    expect(
      out.map((d) => (d.metadata as { filePath: string }).filePath),
    ).toEqual(["src/a.ts", "docs/b.md", ".github/workflows/release.yml"]);
  });

  it("filters by `**/.github/workflows/*.yml` and keeps only matches", async () => {
    const store = stubStore([
      makeDoc("src/a.ts"),
      makeDoc(".github/workflows/release.yml"),
      makeDoc("docs/b.md"),
      makeDoc("foo/.github/workflows/ci.yml"),
      makeDoc(".github/workflows/build.yaml"),
    ]);
    const out = await searchCodebase("x", store as never, {
      limit: 10,
      pathGlob: "**/.github/workflows/*.yml",
    });
    expect(
      out.map((d) => (d.metadata as { filePath: string }).filePath),
    ).toEqual([
      ".github/workflows/release.yml",
      "foo/.github/workflows/ci.yml",
    ]);
  });

  it("composes pathGlob AND repo filter", async () => {
    const store = stubStore([
      makeDoc("src/a.ts", "alpha"),
      makeDoc("src/a.ts", "beta"),
      makeDoc(".github/workflows/release.yml", "alpha"),
      makeDoc(".github/workflows/release.yml", "beta"),
    ]);
    const out = await searchCodebase("x", store as never, {
      limit: 10,
      repo: "alpha",
      pathGlob: "**/release.yml",
    });
    expect(out).toHaveLength(1);
    expect((out[0].metadata as { repo: string; filePath: string }).repo).toBe(
      "alpha",
    );
    expect((out[0].metadata as { filePath: string }).filePath).toBe(
      ".github/workflows/release.yml",
    );
  });

  it("respects limit after the filter", async () => {
    const store = stubStore([
      makeDoc(".github/workflows/a.yml"),
      makeDoc(".github/workflows/b.yml"),
      makeDoc(".github/workflows/c.yml"),
      makeDoc("src/x.ts"),
    ]);
    const out = await searchCodebase("x", store as never, {
      limit: 2,
      pathGlob: "**/.github/workflows/*.yml",
    });
    expect(out).toHaveLength(2);
  });

  it("supports brace alternatives", async () => {
    const store = stubStore([
      makeDoc("src/a.ts"),
      makeDoc("src/a.tsx"),
      makeDoc("src/a.js"),
      makeDoc("docs/a.md"),
    ]);
    const out = await searchCodebase("x", store as never, {
      limit: 10,
      pathGlob: "**/*.{ts,tsx}",
    });
    expect(
      out.map((d) => (d.metadata as { filePath: string }).filePath),
    ).toEqual(["src/a.ts", "src/a.tsx"]);
  });

  it("returns empty when nothing matches the glob", async () => {
    const store = stubStore([makeDoc("src/a.ts"), makeDoc("src/b.ts")]);
    const out = await searchCodebase("x", store as never, {
      limit: 10,
      pathGlob: "**/Dockerfile",
    });
    expect(out).toEqual([]);
  });

  it("finds matches that only surface in the over-fetch window", async () => {
    // Vector store returns 30 noise docs ahead of one true match. With
    // limit=5 the over-fetch is 5 * 4 = 20 — NOT enough to reach the
    // match — which would (correctly) miss it. With limit=10 the over-
    // fetch is 40, well past the match's index of 30, and the result
    // must include it. This pins the multiplier-vs-recall contract so
    // a regression that drops the over-fetch would fail loudly.
    const docs: Document[] = [];
    for (let i = 0; i < 30; i++) {
      docs.push(makeDoc(`src/noise-${i}.ts`));
    }
    docs.push(makeDoc("Dockerfile"));
    const store = stubStore(docs);

    const tooNarrow = await searchCodebase("x", store as never, {
      limit: 5,
      pathGlob: "**/Dockerfile",
    });
    expect(tooNarrow).toEqual([]);

    const wide = await searchCodebase("x", store as never, {
      limit: 10,
      pathGlob: "**/Dockerfile",
    });
    expect(wide).toHaveLength(1);
    expect((wide[0].metadata as { filePath: string }).filePath).toBe(
      "Dockerfile",
    );
  });
});

describe("formatChunkLocation", () => {
  it("renders path:start-end when both line numbers are present", () => {
    expect(
      formatChunkLocation({ filePath: "r/a.ts", lineStart: 1, lineEnd: 30 }),
    ).toBe("r/a.ts:1-30");
  });

  it("renders path:line when start equals end", () => {
    expect(
      formatChunkLocation({ filePath: "r/a.ts", lineStart: 5, lineEnd: 5 }),
    ).toBe("r/a.ts:5");
  });

  it("falls back to bare filePath when line numbers are missing", () => {
    expect(formatChunkLocation({ filePath: "r/a.ts" })).toBe("r/a.ts");
  });

  it("falls back when only one of the two line numbers is present", () => {
    expect(formatChunkLocation({ filePath: "r/a.ts", lineStart: 5 })).toBe(
      "r/a.ts",
    );
    expect(formatChunkLocation({ filePath: "r/a.ts", lineEnd: 9 })).toBe(
      "r/a.ts",
    );
  });
});

// ── OKF frontmatter metadata: type/tags filters, display, pointers ─────────

describe("searchCodebase type/tags filters (stub store)", () => {
  function stubStore(docs: Document[]) {
    return {
      similaritySearch: async (
        _query: string,
        k: number,
        filter?: Record<string, string>,
      ): Promise<Document[]> => {
        const scoped = filter?.repo
          ? docs.filter(
              (d) => (d.metadata as { repo?: string }).repo === filter.repo,
            )
          : docs;
        return scoped.slice(0, k);
      },
    };
  }

  function makeDoc(
    filePath: string,
    metadata: Record<string, unknown> = {},
    repo = "demo",
  ): Document {
    return new Document({
      pageContent: `// ${filePath}`,
      metadata: { filePath, repo, ...metadata },
    });
  }

  it("type filter matches chunks whose fmType strictly equals the value", async () => {
    const store = stubStore([
      makeDoc("a.md", { fmType: "module" }),
      makeDoc("b.md", { fmType: "guide" }),
    ]);
    const out = await searchCodebase("x", store as never, {
      limit: 10,
      type: "module",
    });
    expect(
      out.map((d) => (d.metadata as { filePath: string }).filePath),
    ).toEqual(["a.md"]);
  });

  it("type filter excludes a mismatched fmType", async () => {
    const store = stubStore([makeDoc("a.md", { fmType: "guide" })]);
    const out = await searchCodebase("x", store as never, {
      limit: 10,
      type: "module",
    });
    expect(out).toEqual([]);
  });

  it("tags filter: a single tag matches chunks whose fmTags contains it", async () => {
    const store = stubStore([
      makeDoc("a.md", { fmTags: ["okf", "backend"] }),
      makeDoc("b.md", { fmTags: ["frontend"] }),
    ]);
    const out = await searchCodebase("x", store as never, {
      limit: 10,
      tags: ["okf"],
    });
    expect(
      out.map((d) => (d.metadata as { filePath: string }).filePath),
    ).toEqual(["a.md"]);
  });

  it("tags filter: two tags require ALL to be present (contains-all)", async () => {
    const store = stubStore([
      makeDoc("both.md", { fmTags: ["okf", "backend", "extra"] }),
      makeDoc("one-only.md", { fmTags: ["okf"] }),
    ]);
    const out = await searchCodebase("x", store as never, {
      limit: 10,
      tags: ["okf", "backend"],
    });
    expect(
      out.map((d) => (d.metadata as { filePath: string }).filePath),
    ).toEqual(["both.md"]);
  });

  it("tags filter excludes a chunk missing one of the requested tags", async () => {
    const store = stubStore([makeDoc("a.md", { fmTags: ["okf"] })]);
    const out = await searchCodebase("x", store as never, {
      limit: 10,
      tags: ["okf", "backend"],
    });
    expect(out).toEqual([]);
  });

  it("excludes a no-metadata chunk when type filter is set, includes it when no filter is set", async () => {
    const store = stubStore([
      makeDoc("plain.md"),
      makeDoc("typed.md", { fmType: "module" }),
    ]);

    const filtered = await searchCodebase("x", store as never, {
      limit: 10,
      type: "module",
    });
    expect(
      filtered.map((d) => (d.metadata as { filePath: string }).filePath),
    ).toEqual(["typed.md"]);

    const unfiltered = await searchCodebase("x", store as never, { limit: 10 });
    expect(
      unfiltered.map((d) => (d.metadata as { filePath: string }).filePath),
    ).toEqual(["plain.md", "typed.md"]);
  });

  it("excludes a no-metadata chunk when tags filter is set, includes it when no filter is set", async () => {
    const store = stubStore([
      makeDoc("plain.md"),
      makeDoc("tagged.md", { fmTags: ["okf"] }),
    ]);

    const filtered = await searchCodebase("x", store as never, {
      limit: 10,
      tags: ["okf"],
    });
    expect(
      filtered.map((d) => (d.metadata as { filePath: string }).filePath),
    ).toEqual(["tagged.md"]);

    const unfiltered = await searchCodebase("x", store as never, { limit: 10 });
    expect(
      unfiltered.map((d) => (d.metadata as { filePath: string }).filePath),
    ).toEqual(["plain.md", "tagged.md"]);
  });

  it("AND-composes type with the repo filter", async () => {
    const store = stubStore([
      makeDoc("a.md", { fmType: "module" }, "alpha"),
      makeDoc("a.md", { fmType: "module" }, "beta"),
    ]);
    const out = await searchCodebase("x", store as never, {
      limit: 10,
      repo: "alpha",
      type: "module",
    });
    expect(out).toHaveLength(1);
    expect((out[0].metadata as { repo: string }).repo).toBe("alpha");
  });

  it("AND-composes tags with the repo filter", async () => {
    const store = stubStore([
      makeDoc("a.md", { fmTags: ["okf", "backend"] }, "alpha"),
      makeDoc("a.md", { fmTags: ["okf", "backend"] }, "beta"),
    ]);
    const out = await searchCodebase("x", store as never, {
      limit: 10,
      repo: "alpha",
      tags: ["okf", "backend"],
    });
    expect(out).toHaveLength(1);
    expect((out[0].metadata as { repo: string }).repo).toBe("alpha");
  });

  it("finds a type match that only surfaces in the over-fetch window", async () => {
    const docs: Document[] = [];
    for (let i = 0; i < 30; i++) docs.push(makeDoc(`noise-${i}.md`));
    docs.push(makeDoc("match.md", { fmType: "module" }));
    const store = stubStore(docs);

    const tooNarrow = await searchCodebase("x", store as never, {
      limit: 5,
      type: "module",
    });
    expect(tooNarrow).toEqual([]);

    const wide = await searchCodebase("x", store as never, {
      limit: 10,
      type: "module",
    });
    expect(wide).toHaveLength(1);
    expect((wide[0].metadata as { filePath: string }).filePath).toBe(
      "match.md",
    );
  });
});

describe("searchCodebase type/tags filters (real store)", () => {
  function fakeEmbeddings(dimension = 8): Embeddings {
    function deterministicVector(text: string): number[] {
      const vec = new Array(dimension).fill(0);
      for (let i = 0; i < text.length; i++) {
        vec[i % dimension] += text.charCodeAt(i) / 1000;
      }
      const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
      return vec.map((v) => v / (norm || 1));
    }
    return {
      embedDocuments: async (texts: string[]) => texts.map(deterministicVector),
      embedQuery: async (text: string) => deterministicVector(text),
    } as unknown as Embeddings;
  }

  function directoryConfig(dataDir: string): Config {
    return {
      scanRoot: "/tmp/test",
      dataDir,
      embeddingProvider: "openai",
      llmProvider: "auto",
      ollamaBaseUrl: "http://localhost:11434/v1",
      embeddingModel: "test",
      llmModel: "test",
      vectorStoreType: "directory",
    };
  }

  const tmpDirs: string[] = [];
  afterEach(async () => {
    while (tmpDirs.length > 0) {
      const dir = tmpDirs.pop();
      if (dir) await rm(dir, { recursive: true, force: true });
    }
  });

  it("filters by fmType and fmTags through a real SqliteStore-backed vector store", async () => {
    const dir = await mkdtemp(join(tmpdir(), "oracle-search-filter-"));
    tmpDirs.push(dir);
    const config = directoryConfig(dir);
    const store = await createVectorStore(fakeEmbeddings(), config);

    await store.addDocuments([
      new Document({
        pageContent: "OKF backend doc",
        metadata: {
          repo: "docs",
          filePath: "docs/okf/backend.md",
          fmType: "module",
          fmTags: ["okf", "backend"],
        },
      }),
      new Document({
        pageContent: "OKF frontend doc",
        metadata: {
          repo: "docs",
          filePath: "docs/okf/frontend.md",
          fmType: "guide",
          fmTags: ["okf", "frontend"],
        },
      }),
      new Document({
        pageContent: "plain doc, no frontmatter",
        metadata: { repo: "docs", filePath: "docs/plain.md" },
      }),
    ]);

    const byType = await searchCodebase("doc", store, {
      limit: 10,
      type: "module",
    });
    expect(
      byType.map((d) => (d.metadata as { filePath: string }).filePath),
    ).toEqual(["docs/okf/backend.md"]);

    const byTags = await searchCodebase("doc", store, {
      limit: 10,
      tags: ["okf", "backend"],
    });
    expect(
      byTags.map((d) => (d.metadata as { filePath: string }).filePath),
    ).toEqual(["docs/okf/backend.md"]);

    const noMatch = await searchCodebase("doc", store, {
      limit: 10,
      type: "nonexistent",
    });
    expect(noMatch).toEqual([]);

    store.close();
  });
});

describe("formatChunkTypeTag", () => {
  it("renders a bracketed tag when fmType is a non-empty string", () => {
    expect(formatChunkTypeTag({ fmType: "module" })).toBe("[module]");
  });

  it("returns an empty string when fmType is absent", () => {
    expect(formatChunkTypeTag({})).toBe("");
  });

  it("returns an empty string when fmType is an empty string or the wrong type", () => {
    expect(formatChunkTypeTag({ fmType: "" })).toBe("");
    expect(formatChunkTypeTag({ fmType: 42 })).toBe("");
  });
});

describe("formatChunkSourcesLine", () => {
  it("joins fmSources into a single 'sources: a, b' line", () => {
    expect(formatChunkSourcesLine({ fmSources: ["a.md", "b.md"] })).toBe(
      "sources: a.md, b.md",
    );
  });

  it("returns null when fmSources is absent", () => {
    expect(formatChunkSourcesLine({})).toBeNull();
  });

  it("returns null when fmSources is not an array or has no valid entries", () => {
    expect(formatChunkSourcesLine({ fmSources: "a.md" })).toBeNull();
    expect(formatChunkSourcesLine({ fmSources: [] })).toBeNull();
    expect(formatChunkSourcesLine({ fmSources: [1, ""] })).toBeNull();
  });
});

describe("formatSearchResults", () => {
  it("renders a plain chunk (no fm metadata) byte-identical to the pre-OKF format", () => {
    const docs = [
      new Document({
        pageContent: "function hello() {}",
        metadata: { filePath: "r/a.ts", repo: "r", lineStart: 1, lineEnd: 1 },
      }),
    ];
    const out = formatSearchResults(docs);
    expect(out).toBe("[1] r/a.ts:1 (r):\nfunction hello() {}");
  });

  it("adds a [type] tag to the header when fmType is present", () => {
    const docs = [
      new Document({
        pageContent: "content",
        metadata: {
          filePath: "docs/okf/backend.md",
          repo: "agent-tasks",
          fmType: "module",
        },
      }),
    ];
    const out = formatSearchResults(docs);
    expect(out).toBe(
      "[1] docs/okf/backend.md (agent-tasks) [module]:\ncontent",
    );
  });

  it("adds a 'sources: ...' line after the header when fmSources is present", () => {
    const docs = [
      new Document({
        pageContent: "content",
        metadata: { filePath: "a.md", repo: "r", fmSources: ["src1", "src2"] },
      }),
    ];
    const out = formatSearchResults(docs);
    expect(out).toBe("[1] a.md (r):\nsources: src1, src2\ncontent");
  });

  it("combines the [type] tag and sources line", () => {
    const docs = [
      new Document({
        pageContent: "content",
        metadata: {
          filePath: "a.md",
          repo: "r",
          fmType: "module",
          fmSources: ["src1"],
        },
      }),
    ];
    const out = formatSearchResults(docs);
    expect(out).toBe("[1] a.md (r) [module]:\nsources: src1\ncontent");
  });

  it("joins multiple chunks with the existing '---' separator", () => {
    const docs = [
      new Document({
        pageContent: "one",
        metadata: { filePath: "a.md", repo: "r" },
      }),
      new Document({
        pageContent: "two",
        metadata: { filePath: "b.md", repo: "r" },
      }),
    ];
    const out = formatSearchResults(docs);
    expect(out).toBe("[1] a.md (r):\none\n\n---\n\n[2] b.md (r):\ntwo");
  });

  it("returns 'No results found.' for an empty doc list", () => {
    expect(formatSearchResults([])).toBe("No results found.");
  });
});

describe("extractSourcePointers", () => {
  it("returns [] when no retrieved chunk has fmSources", () => {
    const docs = [
      new Document({
        pageContent: "a",
        metadata: { filePath: "a.md", repo: "r" },
      }),
      new Document({
        pageContent: "b",
        metadata: { filePath: "b.md", repo: "r" },
      }),
    ];
    expect(extractSourcePointers(docs)).toEqual([]);
  });

  it("dedupes across chunks and orders by first appearance in retrieval rank", () => {
    const docs = [
      new Document({
        pageContent: "a",
        metadata: { filePath: "a.md", repo: "r", fmSources: ["src2", "src1"] },
      }),
      new Document({
        pageContent: "b",
        metadata: { filePath: "b.md", repo: "r", fmSources: ["src1", "src3"] },
      }),
    ];
    expect(extractSourcePointers(docs)).toEqual(["src2", "src1", "src3"]);
  });

  it("ignores non-array fmSources and non-string/empty entries", () => {
    const docs = [
      new Document({
        pageContent: "a",
        metadata: { filePath: "a.md", repo: "r", fmSources: "x" },
      }),
      new Document({
        pageContent: "b",
        metadata: { filePath: "b.md", repo: "r", fmSources: [1, "", "valid"] },
      }),
    ];
    expect(extractSourcePointers(docs)).toEqual(["valid"]);
  });
});

describe("formatPointersSection", () => {
  it("returns '' (omitted) when there are no pointers", () => {
    expect(formatPointersSection([])).toBe("");
  });

  it("renders the exact label and one line per path when under the cap", () => {
    const out = formatPointersSection(["a.md", "b.md"]);
    expect(out).toBe(
      "\n\nPointers (from OKF sources metadata):\n- a.md\n- b.md",
    );
  });

  it("caps at 10 entries and appends a truncation note", () => {
    const pointers = Array.from({ length: 13 }, (_, i) => `src${i}.md`);
    const out = formatPointersSection(pointers);
    const lines = out.split("\n");
    // Header line is "" then the label (leading \n\n split gives ["", "", label, ...]).
    expect(out).toContain("Pointers (from OKF sources metadata):");
    for (let i = 0; i < 10; i++) {
      expect(out).toContain(`- src${i}.md`);
    }
    expect(out).not.toContain("- src10.md");
    expect(out).toContain("... and 3 more");
    expect(lines[lines.length - 1]).toBe("... and 3 more");
  });

  it("does not truncate exactly at the cap boundary (10 items, no note)", () => {
    const pointers = Array.from({ length: 10 }, (_, i) => `src${i}.md`);
    const out = formatPointersSection(pointers);
    expect(out).not.toContain("more");
  });
});

describe("parseCommaSeparatedList", () => {
  it("splits a comma-separated string and trims whitespace", () => {
    expect(parseCommaSeparatedList("a, b,c")).toEqual(["a", "b", "c"]);
  });

  it("returns undefined for undefined input", () => {
    expect(parseCommaSeparatedList(undefined)).toBeUndefined();
  });

  it("returns undefined for an empty string", () => {
    expect(parseCommaSeparatedList("")).toBeUndefined();
  });

  it("drops empty segments from trailing/consecutive commas", () => {
    expect(parseCommaSeparatedList("a,,b,")).toEqual(["a", "b"]);
  });

  it("returns undefined when every segment is empty after trimming", () => {
    expect(parseCommaSeparatedList(" , ,")).toBeUndefined();
  });
});
