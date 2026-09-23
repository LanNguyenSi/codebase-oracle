import { describe, it, expect, afterEach } from "vitest";
import { Document } from "@langchain/core/documents";
import { createServer, type Server } from "node:net";
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
    llmTimeoutMs: 120_000,
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
      // No LLM configured (auto, no keys) is not the LLM-failure branch:
      // degraded stays unset.
      expect(result.degraded).toBeUndefined();
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
    // Machine-readable degraded marker for CLI --json consumers.
    expect(result.degraded).toBe(true);
    expect(result.degradedReason).toBe("llm_request_failed");
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
    // A successful LLM answer is not degraded.
    expect(result.degraded).toBeUndefined();
  });
});

// ── Bounded LLM timeout (task 844aac2c) ───────────────────────────────────
//
// A real (no mock) LLM constructed through the openai-compatible lane,
// pointed at a local TCP server that ACCEPTS the connection and then never
// writes a byte back (no HTTP response at all, the closed-port case from
// the task ("a closed local port took more than 70 s to surface as a
// failure") plus the slow-but-connected case a closed port alone doesn't
// cover). Without a request timeout this hangs indefinitely; with
// config.llmTimeoutMs set, the underlying HTTP client aborts the request
// after that bound and queryCodebase's existing catch-and-degrade branch
// (exercised via deps injection above) takes over for real, through the
// full createLlm -> chain.invoke() path with no injection seam at all.

describe("queryCodebase, real LLM client against a black-hole TCP server (task 844aac2c)", () => {
  let server: Server | undefined;
  // The client's socket is deliberately never closed by the black-hole
  // handler below (that's the point: it never writes or ends), so
  // server.close() alone would hang waiting for it to drain. Track and
  // destroy every accepted socket on teardown instead.
  let sockets: Array<{ destroy(): void }>;

  afterEach(async () => {
    if (!server) return;
    for (const socket of sockets) socket.destroy();
    await new Promise<void>((resolve) => server!.close(() => resolve()));
    server = undefined;
  });

  it("returns degraded:true within the configured llmTimeoutMs bound instead of hanging", async () => {
    sockets = [];
    // Accept every connection and never write or end it: the client's
    // socket connects successfully, so this is a response-timeout (not a
    // connection-refused) scenario, and never resolves on its own.
    server = createServer((socket) => {
      sockets.push(socket);
      socket.on("error", () => {
        // Ignore ECONNRESET from the client's own abort-on-timeout, or from
        // the teardown-time destroy() above; a black-hole server that
        // crashed on either would be a test bug, not a passing run.
      });
    });
    const port = await new Promise<number>((resolve, reject) => {
      server!.on("error", reject);
      server!.listen(0, "127.0.0.1", () => {
        const address = server!.address();
        if (address === null || typeof address === "string") {
          reject(new Error("expected a bound TCP port"));
          return;
        }
        resolve(address.port);
      });
    });

    const boundMs = 300;
    const config = baseConfig({
      llmProvider: "openai-compatible",
      llmBaseUrl: `http://127.0.0.1:${port}`,
      llmApiKey: "test-key",
      llmModel: "test-model",
      llmTimeoutMs: boundMs,
    });
    const docs = [
      new Document({
        pageContent: "function hello() {}",
        metadata: { filePath: "src/a.ts", repo: "repo-a" },
      }),
    ];
    const store: VectorStoreWrapper = {
      similaritySearch: async () => docs,
      addDocuments: async () => {},
      listRepos: () => [],
      getFileMetadata: () => null,
      getFirstChunkByFile: () => null,
      close: () => {},
    };

    const startedAt = Date.now();
    // No `deps` override: this goes through the real `createLlm` (as
    // `queryCodebase`'s own default) and the real ChatOpenAI client, so the
    // timeout under test is the one actually wired into the constructor,
    // not a stand-in for it.
    const result = await queryCodebase("what does hello() do?", store, config);
    const elapsedMs = Date.now() - startedAt;

    expect(result.degraded).toBe(true);
    expect(result.degradedReason).toBe("llm_request_failed");
    expect(result.answer).toMatch(/^LLM request failed/);
    // Lower bound: a `timeout: 1` mutant (or any bound effectively disabled)
    // would return near-instantly instead of actually waiting out boundMs,
    // so this by itself discriminates that class of mutant. 50ms slack for
    // timer-firing jitter.
    expect(elapsedMs).toBeGreaterThanOrEqual(boundMs - 50);
    // Upper bound, generous slack (server-side accept latency, event-loop
    // scheduling) but still tight enough that an unbounded/very-long timeout
    // or a disabled timeout (which would hang until the test's own runner
    // timeout, tens of seconds away) fails this assertion instead of
    // passing by luck.
    expect(elapsedMs).toBeLessThan(boundMs + 4_000);
  }, 15_000);
});

// ── Invoke-level overall deadline (task 844aac2c, round 2) ────────────────
//
// The constructor-level `timeout` set on each LLM client (exercised by the
// black-hole-server test above) only bounds time-to-first-response-byte for
// the OpenAI/Anthropic SDKs (see docs/configuration.md and the comment on
// `LLM_MAX_RETRIES` in chain.ts). A server that ACCEPTS the connection,
// sends response headers promptly, and then stalls the body indefinitely
// never trips that bound. This exercises the separate invoke-level deadline
// (`chain.invoke(..., { timeout: config.llmTimeoutMs })`) that covers the
// whole non-streaming call, headers-received or not.

describe("queryCodebase, real LLM client against a headers-then-stall TCP server (task 844aac2c, round 2)", () => {
  let server: Server | undefined;
  let sockets: Array<{ destroy(): void }>;

  afterEach(async () => {
    if (!server) return;
    for (const socket of sockets) socket.destroy();
    await new Promise<void>((resolve) => server!.close(() => resolve()));
    server = undefined;
  });

  it("returns degraded:true within the configured llmTimeoutMs bound even after response headers arrive", async () => {
    sockets = [];
    // Send a complete, valid HTTP status line and headers immediately (so
    // the constructor-level "time to first byte" timeout is satisfied and
    // would NOT fire), declare a chunked body, then never write a single
    // body chunk or terminator: the response never completes.
    server = createServer((socket) => {
      sockets.push(socket);
      socket.on("error", () => {
        // Ignore ECONNRESET from the client's own abort-on-timeout, or from
        // the teardown-time destroy() above.
      });
      socket.write(
        "HTTP/1.1 200 OK\r\n" +
          "Content-Type: application/json\r\n" +
          "Transfer-Encoding: chunked\r\n" +
          "\r\n",
      );
      // Deliberately no further writes: the body stalls forever.
    });
    const port = await new Promise<number>((resolve, reject) => {
      server!.on("error", reject);
      server!.listen(0, "127.0.0.1", () => {
        const address = server!.address();
        if (address === null || typeof address === "string") {
          reject(new Error("expected a bound TCP port"));
          return;
        }
        resolve(address.port);
      });
    });

    const boundMs = 300;
    const config = baseConfig({
      llmProvider: "openai-compatible",
      llmBaseUrl: `http://127.0.0.1:${port}`,
      llmApiKey: "test-key",
      llmModel: "test-model",
      llmTimeoutMs: boundMs,
    });
    const docs = [
      new Document({
        pageContent: "function hello() {}",
        metadata: { filePath: "src/a.ts", repo: "repo-a" },
      }),
    ];
    const store: VectorStoreWrapper = {
      similaritySearch: async () => docs,
      addDocuments: async () => {},
      listRepos: () => [],
      getFileMetadata: () => null,
      getFirstChunkByFile: () => null,
      close: () => {},
    };

    const startedAt = Date.now();
    const result = await queryCodebase("what does hello() do?", store, config);
    const elapsedMs = Date.now() - startedAt;

    expect(result.degraded).toBe(true);
    expect(result.degradedReason).toBe("llm_request_failed");
    expect(result.answer).toMatch(/^LLM request failed/);
    // Lower bound discriminates a mutant that drops or disables the
    // invoke-level timeout (the response would otherwise never resolve
    // within the test's own runner timeout).
    expect(elapsedMs).toBeGreaterThanOrEqual(boundMs - 50);
    expect(elapsedMs).toBeLessThan(boundMs + 4_000);
  }, 15_000);
});
