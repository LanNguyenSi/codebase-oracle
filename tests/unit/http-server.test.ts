/**
 * Tests for the http-server.ts auth gate, health check, and 404 routing.
 *
 * The module-level guard added in this task means importing http-server.ts
 * no longer binds a port or calls process.exit(1). Tests call
 * createHttpRequestHandler() directly, create a node:http.Server, and listen
 * on port 0 (OS-assigned ephemeral port) so multiple parallel test runs
 * never collide.
 *
 * Gap covered: auth-gate 401, WWW-Authenticate header, -32001 JSON body,
 * GET /health → 200, unknown path → 404, the post-auth 500 error mapping
 * (IndexFingerprintError → its message; any other throw → generic "Internal
 * error"), and the per-request releasePair() guard (transport/server closed
 * on the response stream's "close" and "error" events).
 *
 * The 500-mapping and releasePair tests drive their failure by mocking
 * WebStandardStreamableHTTPServerTransport.prototype.handleRequest for a
 * single call (vi.spyOn + mockImplementationOnce), rather than mocking the
 * store: a real IndexFingerprintError thrown by a tool handler is caught and
 * turned into a normal JSON-RPC error response *inside* the MCP SDK's
 * request dispatch (transport.onmessage is fire-and-forget, not awaited by
 * handlePostRequest), so it never reaches createHttpRequestHandler's own
 * try/catch. Making handleRequest itself throw/return a controllable stream
 * exercises that outer catch block and the stream-release listeners
 * directly and deterministically.
 */
import { createServer } from "node:http";
import {
  afterAll,
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";
import { Document } from "@langchain/core/documents";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { WebStandardStreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/webStandardStreamableHttp.js";
import type { HttpBindConfig } from "../../src/http-auth.js";
import { VERSION } from "../../src/version.js";

// http-server.ts computes its module-level `config` via loadConfig() at
// import time (below), reading real process.env. Pin the embedding provider
// to the repo's deterministic in-process stub (src/store/embeddings.ts,
// also used by tests/integration/index-cli.test.ts) BEFORE that import
// evaluates, via vi.hoisted: vitest hoists this above every import in the
// file, including the static `import ... from "../../src/http-server.js"`
// below, so loadConfig() never sees the real (or absent) OPENAI_API_KEY /
// ORACLE_EMBEDDING_PROVIDER. Without this, createEmbeddings() takes the real
// OpenAI path and throws "OPENAI_API_KEY is required..." whenever this file
// runs without ambient env (e.g. CI, which has no .env and no such secret),
// so this file must not depend on another test file, a developer's local
// .env, or CI job/worker ordering for that. vi.stubEnv is undone in the
// afterAll below so the mutation never leaks to a test file that runs after
// this one in the same worker process (process.env is a real, shared
// Node.js global, not something vitest's per-file module isolation resets).
vi.hoisted(() => {
  vi.stubEnv("ORACLE_EMBEDDING_PROVIDER", "stub");
});

// vi.mock is hoisted above these imports. We keep the real IndexFingerprintError
// class (and everything else) via importOriginal, only replacing
// createVectorStore so the new oracle_search pass-through test below can drive
// a fake store without a real DB/embeddings/network. None of the OTHER tests
// in this file exercise any MCP tool (they only hit auth/health/404 routing),
// so this mock has no effect on them.
vi.mock("../../src/store/vector-store.js", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("../../src/store/vector-store.js")>();
  return { ...actual, createVectorStore: vi.fn() };
});

import {
  createHttpRequestHandler,
  buildServer,
} from "../../src/http-server.js";
import {
  createVectorStore,
  IndexFingerprintError,
} from "../../src/store/vector-store.js";
import type { VectorStoreWrapper } from "../../src/store/vector-store.js";

// Undo the vi.stubEnv from the vi.hoisted block above once every test in
// this file has run, so the stubbed ORACLE_EMBEDDING_PROVIDER never leaks
// into a different test file sharing the same worker process.
afterAll(() => {
  vi.unstubAllEnvs();
});

// Safety net on top of the existing per-test finally blocks: restores any
// vi.spyOn mock left behind by a failed assertion (which would otherwise
// skip its finally's mockRestore()) so a spy from one test never leaks into
// the next.
afterEach(() => {
  vi.restoreAllMocks();
});

const TOKEN = "test-bearer-token-abc123";
const LOOPBACK_WITH_TOKEN: HttpBindConfig = { bind: "127.0.0.1", token: TOKEN };
const LOOPBACK_NO_TOKEN: HttpBindConfig = { bind: "127.0.0.1", token: null };

/** Spin up an http.Server on an ephemeral port; returns [url, closeFn]. */
function startServer(bindConfig: HttpBindConfig, port = 3100) {
  const handler = createHttpRequestHandler(bindConfig, port);
  const server = createServer(handler);
  return new Promise<{ url: string; close: () => Promise<void> }>((resolve) => {
    server.listen(0, "127.0.0.1", () => {
      const addr = server.address();
      const p = typeof addr === "object" && addr ? addr.port : 0;
      resolve({
        url: `http://127.0.0.1:${p}`,
        close: () =>
          new Promise<void>((res, rej) =>
            server.close((err) => (err ? rej(err) : res())),
          ),
      });
    });
  });
}

describe("createHttpRequestHandler — auth gate (token configured)", () => {
  let url: string;
  let close: () => Promise<void>;

  beforeEach(async () => {
    ({ url, close } = await startServer(LOOPBACK_WITH_TOKEN));
  });

  afterEach(async () => {
    await close();
  });

  it("POST /mcp with no Authorization header → 401 with WWW-Authenticate and -32001 JSON", async () => {
    const res = await fetch(`${url}/mcp`, { method: "POST" });
    expect(res.status).toBe(401);
    expect(res.headers.get("www-authenticate")).toMatch(/Bearer/i);
    const body = (await res.json()) as {
      jsonrpc: string;
      error: { code: number; message: string };
      id: null;
    };
    expect(body.jsonrpc).toBe("2.0");
    expect(body.error.code).toBe(-32001);
    expect(body.error.message).toContain("Unauthorized");
    expect(body.error.message).toContain("missing");
    expect(body.id).toBeNull();
  });

  it("POST /mcp with the wrong token → 401", async () => {
    const res = await fetch(`${url}/mcp`, {
      method: "POST",
      headers: { Authorization: "Bearer wrong-token" },
    });
    expect(res.status).toBe(401);
    const body = (await res.json()) as {
      error: { code: number; message: string };
    };
    expect(body.error.code).toBe(-32001);
    expect(body.error.message).toContain("invalid");
  });

  it("POST /mcp with the correct token → NOT 401 (passes the auth gate)", async () => {
    // We cannot easily make the MCP layer succeed without a real index,
    // but the point here is that the auth gate passes (status is not 401).
    // The transport/server will return some non-401 status once the body
    // is processed (200 or 500 depending on the payload).
    const res = await fetch(`${url}/mcp`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${TOKEN}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        jsonrpc: "2.0",
        id: 1,
        method: "initialize",
        params: {
          protocolVersion: "2024-11-05",
          capabilities: {},
          clientInfo: { name: "test", version: "1.0" },
        },
      }),
    });
    expect(res.status).not.toBe(401);
  });

  // Mutation guard: if we invert the `if (!auth.ok)` guard, the 401 tests fail.
  // Documented here: inverting would return 404 (falls through to the
  // "everything else" branch) or 200, which would fail the status assertions above.
});

describe("createHttpRequestHandler — GET /health", () => {
  let url: string;
  let close: () => Promise<void>;

  beforeEach(async () => {
    ({ url, close } = await startServer(LOOPBACK_NO_TOKEN));
  });

  afterEach(async () => {
    await close();
  });

  it("GET /health → 200 with { status: 'ok', version } JSON body", async () => {
    const res = await fetch(`${url}/health`);
    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toContain("application/json");
    const body = (await res.json()) as { status: string; version: string };
    expect(body.status).toBe("ok");
    expect(body.version).toBe(VERSION);
  });
});

describe("createHttpRequestHandler — unknown paths → 404", () => {
  let url: string;
  let close: () => Promise<void>;

  beforeEach(async () => {
    ({ url, close } = await startServer(LOOPBACK_NO_TOKEN));
  });

  afterEach(async () => {
    await close();
  });

  it("GET / → 404", async () => {
    const res = await fetch(`${url}/`);
    expect(res.status).toBe(404);
  });

  it("GET /unknown → 404 with error JSON", async () => {
    const res = await fetch(`${url}/unknown`);
    expect(res.status).toBe(404);
    const body = (await res.json()) as { error: string };
    expect(body.error).toContain("Not found");
  });

  it("POST /unknown → 404", async () => {
    const res = await fetch(`${url}/unknown`, { method: "POST" });
    expect(res.status).toBe(404);
  });
});

// ── post-auth 500 error mapping ───────────────────────────────────────────
//
// Forces the try/catch in createHttpRequestHandler's POST /mcp branch by
// making WebStandardStreamableHTTPServerTransport.prototype.handleRequest
// throw for exactly one call (vi.spyOn + mockImplementationOnce, restored in
// a finally). Auth passes first (LOOPBACK_NO_TOKEN), then the throw happens
// on the awaited `transport.handleRequest(request)` call, landing in the
// catch block that maps IndexFingerprintError -> its message and anything
// else -> the generic "Internal error" (never leaking the real message).
describe("createHttpRequestHandler — post-auth 500 error mapping", () => {
  let url: string;
  let close: () => Promise<void>;

  beforeEach(async () => {
    ({ url, close } = await startServer(LOOPBACK_NO_TOKEN));
  });

  afterEach(async () => {
    await close();
  });

  it("IndexFingerprintError from the MCP handler path → 500 with the error's own message, and releases the per-request transport/server pair", async () => {
    const errSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    const transportCloseSpy = vi.spyOn(
      WebStandardStreamableHTTPServerTransport.prototype,
      "close",
    );
    const serverCloseSpy = vi.spyOn(McpServer.prototype, "close");
    const handleRequestSpy = vi
      .spyOn(WebStandardStreamableHTTPServerTransport.prototype, "handleRequest")
      .mockImplementationOnce(async () => {
        throw new IndexFingerprintError(
          "index fingerprint mismatch: rebuild required",
        );
      });
    try {
      const res = await fetch(`${url}/mcp`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ jsonrpc: "2.0", id: 1, method: "ping" }),
      });
      expect(res.status).toBe(500);
      const body = (await res.json()) as {
        jsonrpc: string;
        error: { code: number; message: string };
        id: null;
      };
      expect(body.jsonrpc).toBe("2.0");
      expect(body.error.code).toBe(-32603);
      expect(body.error.message).toBe(
        "index fingerprint mismatch: rebuild required",
      );
      expect(body.id).toBeNull();
      // Pins that the server also logs the error server-side (not just
      // maps it to a client-facing response).
      expect(errSpy).toHaveBeenCalled();
      // The throw happens on the awaited transport.handleRequest() call,
      // i.e. AFTER server.connect(transport) already created and connected
      // the per-request pair. The catch block must release it (exactly
      // once each — no double-release) or every 500 leaks a connected
      // transport/server pair.
      expect(transportCloseSpy).toHaveBeenCalledTimes(1);
      expect(serverCloseSpy).toHaveBeenCalledTimes(1);
    } finally {
      handleRequestSpy.mockRestore();
      errSpy.mockRestore();
      transportCloseSpy.mockRestore();
      serverCloseSpy.mockRestore();
    }
  });

  it("a generic (non-IndexFingerprintError) throw from the MCP handler path → 500 with a generic 'Internal error' message (no internal detail leak), and releases the per-request transport/server pair", async () => {
    const errSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    const transportCloseSpy = vi.spyOn(
      WebStandardStreamableHTTPServerTransport.prototype,
      "close",
    );
    const serverCloseSpy = vi.spyOn(McpServer.prototype, "close");
    const handleRequestSpy = vi
      .spyOn(WebStandardStreamableHTTPServerTransport.prototype, "handleRequest")
      .mockImplementationOnce(async () => {
        throw new Error("some internal detail that must not leak to clients");
      });
    try {
      const res = await fetch(`${url}/mcp`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ jsonrpc: "2.0", id: 1, method: "ping" }),
      });
      expect(res.status).toBe(500);
      const body = (await res.json()) as {
        error: { code: number; message: string };
      };
      expect(body.error.code).toBe(-32603);
      expect(body.error.message).toBe("Internal error");
      expect(body.error.message).not.toContain("some internal detail");
      // Pins that the server also logs the error server-side (not just
      // maps it to a client-facing response).
      expect(errSpy).toHaveBeenCalled();
      // Same leak guard as the IndexFingerprintError case above, exercised
      // via the generic-throw arm instead.
      expect(transportCloseSpy).toHaveBeenCalledTimes(1);
      expect(serverCloseSpy).toHaveBeenCalledTimes(1);
    } finally {
      handleRequestSpy.mockRestore();
      errSpy.mockRestore();
      transportCloseSpy.mockRestore();
      serverCloseSpy.mockRestore();
    }
  });

  it("a throw that lands in the catch AFTER res.writeHead() already sent headers does not attempt a second writeHead, and still releases the pair", async () => {
    // Drives the `if (!res.headersSent)` branch's FALSE arm: res.writeHead()
    // runs first (forwarding the mocked response's real status/headers),
    // then Readable.fromWeb(response.body) throws synchronously because
    // `body` here is not a real ReadableStream (Node: "argument must be an
    // instance of ReadableStream") — landing in the SAME outer catch as the
    // 500-mapping tests above, but this time with headersSent already true.
    // Node's http server surfaces an attempted second writeHead as a
    // "Cannot set headers after they are sent" throw; the observable proof
    // this test pins is that the ORIGINAL headers (200, text/event-stream)
    // reach the client rather than a 500, since node:http silently keeps
    // the already-sent status/headers when a later writeHead is skipped.
    const errSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    const transportCloseSpy = vi.spyOn(
      WebStandardStreamableHTTPServerTransport.prototype,
      "close",
    );
    const serverCloseSpy = vi.spyOn(McpServer.prototype, "close");
    const handleRequestSpy = vi
      .spyOn(WebStandardStreamableHTTPServerTransport.prototype, "handleRequest")
      .mockImplementationOnce(async () => {
        return {
          status: 200,
          headers: new Headers({ "Content-Type": "text/event-stream" }),
          // Not a real ReadableStream: Readable.fromWeb(body) throws
          // synchronously, AFTER res.writeHead() already ran above it.
          body: {},
        } as unknown as Response;
      });
    try {
      const res = await fetch(`${url}/mcp`, { method: "POST", body: "{}" });
      expect(res.status).toBe(200);
      expect(res.headers.get("content-type")).toBe("text/event-stream");
      // The server-side error is still logged even though headers were
      // already sent.
      expect(errSpy).toHaveBeenCalled();
      // The pair was already created and connected before the throw, so
      // the catch block must still release it exactly once each.
      expect(transportCloseSpy).toHaveBeenCalledTimes(1);
      expect(serverCloseSpy).toHaveBeenCalledTimes(1);
    } finally {
      handleRequestSpy.mockRestore();
      errSpy.mockRestore();
      transportCloseSpy.mockRestore();
      serverCloseSpy.mockRestore();
    }
  });

  // Mutation guard: the two tests above pin both arms of the
  // `err instanceof IndexFingerprintError ? err.message : "Internal error"`
  // ternary in src/http-server.ts. Verified by scratch mutation (see task
  // report): hardcoding the message to "Internal error" fails the first
  // test; dropping the instanceof check (always using err.message) fails
  // the second test by leaking the internal detail.
});

// ── per-request release guard (releasePair) ───────────────────────────────
//
// src/http-server.ts wires releasePair() (closes the per-request transport
// and McpServer) to both the "close" and "error" events of the Node stream
// wrapping the MCP response body, so a per-request transport/server pair is
// never left dangling. releasePair() itself is guarded against
// double-invocation (a `released` flag), so it closes the transport/server
// at most once per request even though "error" and the subsequent "close"
// (Node's autoDestroy emits both) can each call it. Rather than relying on
// a real MCP tool call and race conditions in a real client abort to
// produce a stream error, this mocks handleRequest to return a Response
// whose body is a ReadableStream under direct test control: one scenario
// ends the stream cleanly (controller.close()), the other fails it
// mid-stream (controller.error(...)) — both counts (1 close/server-close
// call each) were confirmed empirically before being pinned here.
describe("createHttpRequestHandler — per-request release guard (releasePair)", () => {
  let url: string;
  let close: () => Promise<void>;

  beforeEach(async () => {
    ({ url, close } = await startServer(LOOPBACK_NO_TOKEN));
  });

  afterEach(async () => {
    await close();
  });

  it("closes the per-request transport and server once the response stream ends cleanly", async () => {
    const transportCloseSpy = vi.spyOn(
      WebStandardStreamableHTTPServerTransport.prototype,
      "close",
    );
    const serverCloseSpy = vi.spyOn(McpServer.prototype, "close");
    const handleRequestSpy = vi
      .spyOn(WebStandardStreamableHTTPServerTransport.prototype, "handleRequest")
      .mockImplementationOnce(async () => {
        const body = new ReadableStream<Uint8Array>({
          start(controller) {
            controller.enqueue(new TextEncoder().encode("data: ok\n\n"));
            controller.close();
          },
        });
        return new Response(body, {
          status: 200,
          headers: { "Content-Type": "text/event-stream" },
        });
      });
    try {
      const res = await fetch(`${url}/mcp`, { method: "POST", body: "{}" });
      await res.text();
      // The stream's "close" event fires asynchronously after the body is
      // fully drained; poll until the listener has run instead of a fixed
      // sleep.
      await vi.waitFor(() =>
        expect(transportCloseSpy).toHaveBeenCalledTimes(1),
      );
      expect(transportCloseSpy).toHaveBeenCalledTimes(1);
      expect(serverCloseSpy).toHaveBeenCalledTimes(1);
    } finally {
      handleRequestSpy.mockRestore();
      transportCloseSpy.mockRestore();
      serverCloseSpy.mockRestore();
    }
  });

  it("tears the client connection down (instead of hanging) and releases the pair once, when the response stream errors mid-flight", async () => {
    // Regression probe: before this task, Readable.fromWeb(...).pipe(res)
    // never forwarded the source stream's "error" to the response, so
    // res.end() was never reached and the client connection stayed open
    // until its own timeout (a probed fetch hung >2000ms, per the task
    // spec). res.destroy(err) in the stream's "error" listener now tears
    // the connection down, so reading the body must settle (reject, since
    // the stream failed mid-flight) well inside a bounded wait instead of
    // hanging — replacing the previous abandon-the-fetch-and-abort-it
    // documentation of that hang.
    const transportCloseSpy = vi.spyOn(
      WebStandardStreamableHTTPServerTransport.prototype,
      "close",
    );
    const serverCloseSpy = vi.spyOn(McpServer.prototype, "close");
    const handleRequestSpy = vi
      .spyOn(WebStandardStreamableHTTPServerTransport.prototype, "handleRequest")
      .mockImplementationOnce(async () => {
        const body = new ReadableStream<Uint8Array>({
          start(controller) {
            controller.enqueue(new TextEncoder().encode("data: partial\n\n"));
            // Simulate a mid-response failure (e.g. the store connection
            // dropping) instead of a clean close.
            controller.error(new Error("simulated stream failure"));
          },
        });
        return new Response(body, {
          status: 200,
          headers: { "Content-Type": "text/event-stream" },
        });
      });
    try {
      // Race the WHOLE request/response cycle (not just body reading)
      // against a 2000ms bound: destroying the response after headers were
      // sent can also surface as fetch() itself rejecting (the underlying
      // socket closing before the client finishes parsing), rather than
      // fetch() resolving and res.text() rejecting — both are an
      // acceptable "settled, did not hang" outcome for this probe.
      const settled: "resolved" | "rejected" = await Promise.race([
        (async () => {
          const res = await fetch(`${url}/mcp`, { method: "POST", body: "{}" });
          try {
            await res.text();
            return "resolved" as const;
          } catch {
            return "rejected" as const;
          }
        })().catch(() => "rejected" as const),
        new Promise<never>((_resolve, reject) =>
          setTimeout(
            () =>
              reject(
                new Error(
                  "the client request did not settle within 2000ms — connection hung",
                ),
              ),
            2000,
          ),
        ),
      ]);
      expect(["resolved", "rejected"]).toContain(settled);
      // releasePair() is guarded against double-invocation, so even though
      // both the "error" listener and the subsequent "close" (Node's
      // autoDestroy emits both) call it, the transport/server are each
      // closed exactly once.
      await vi.waitFor(() =>
        expect(transportCloseSpy).toHaveBeenCalledTimes(1),
      );
      expect(transportCloseSpy).toHaveBeenCalledTimes(1);
      expect(serverCloseSpy).toHaveBeenCalledTimes(1);
    } finally {
      handleRequestSpy.mockRestore();
      transportCloseSpy.mockRestore();
      serverCloseSpy.mockRestore();
    }
  });

  // Mutation guard: removing the `.on("close", releasePair)` registration in
  // src/http-server.ts fails the clean-end test (0 calls instead of 1).
  // Removing the `res.destroy(err)` call from the "error" listener fails
  // the mid-flight-error test above: releasePair() still runs, but the
  // client connection is never torn down, so res.text() hangs past the
  // 2000ms bound instead of rejecting.
});

// ── oracle_search type/tags pass-through (OKF metadata) + sources-expansion ──
//
// Connects an MCP client directly to buildServer() via InMemoryTransport,
// bypassing the HTTP transport entirely (mirrors mcp-server.test.ts's
// connectClient() pattern). createVectorStore is mocked so no real DB or
// embeddings are touched. Kept to a single test with multiple calls sharing
// one connection: http-server.ts's getStore()/storePromise cache is
// module-level (not per-instance like mcp-server.ts's createMcpServer()), so
// splitting this across multiple `it` blocks would let a later test observe
// the earlier test's cached store instead of its own mock — the FIRST tool
// call in this file to actually invoke createVectorStore() wins the cache for
// every subsequent test in the file. The sources-expansion assertions (which
// mirror mcp-server.test.ts's two oracle_search expand_sources tests) are
// folded into this same test for exactly that reason: the store is
// query-dispatching so one instance serves both scenarios.
describe("oracle_search type/tags pass-through (HTTP MCP)", () => {
  function fakeStore(
    similaritySearch: VectorStoreWrapper["similaritySearch"],
    getFirstChunkByFile: VectorStoreWrapper["getFirstChunkByFile"] = () =>
      null,
  ): VectorStoreWrapper {
    return {
      addDocuments: async () => {},
      similaritySearch,
      listRepos: () => [],
      getFileMetadata: () => null,
      getFirstChunkByFile,
      close: () => {},
    };
  }

  it("passes `type` and `tags` through to the real filtering logic, and sources-expansion respects expand_sources", async () => {
    const docs = [
      new Document({
        pageContent: "backend doc",
        metadata: {
          repo: "docs",
          filePath: "docs/okf/backend.md",
          fmType: "module",
          fmTags: ["okf", "backend"],
        },
      }),
      new Document({
        pageContent: "frontend doc",
        metadata: {
          repo: "docs",
          filePath: "docs/okf/frontend.md",
          fmType: "guide",
        },
      }),
    ];
    const moduleDoc = new Document({
      pageContent: "the doc body",
      metadata: {
        repo: "docs",
        filePath: "docs/okf/module.md",
        fmSources: ["src/impl.ts"],
      },
    });
    const implChunk = {
      pageContent: "impl body",
      metadata: { repo: "docs", filePath: "src/impl.ts" },
    };
    const getFirstChunkByFile = vi.fn((repo: string, filePath: string) =>
      repo === "docs" && filePath === "src/impl.ts" ? implChunk : null,
    );
    vi.mocked(createVectorStore).mockResolvedValue(
      fakeStore(async (query: string) => {
        // Query-dispatching so this single cached store instance can also
        // serve the sources-expansion assertions below (see module comment).
        if (query === "module") return [moduleDoc];
        return docs;
      }, getFirstChunkByFile),
    );

    const server = buildServer();
    const [clientTransport, serverTransport] =
      InMemoryTransport.createLinkedPair();
    await server.connect(serverTransport);
    const client = new Client(
      { name: "test-client", version: "1.0.0" },
      { capabilities: {} },
    );
    await client.connect(clientTransport);

    try {
      // Tool schema exposes the new params.
      const { tools } = await client.listTools();
      const searchTool = tools.find((t) => t.name === "oracle_search");
      const props = searchTool?.inputSchema.properties as Record<
        string,
        { type?: string }
      >;
      expect(props).toHaveProperty("type");
      expect(props).toHaveProperty("tags");

      // type filter round-trips.
      const byType = await client.callTool({
        name: "oracle_search",
        arguments: { query: "doc", type: "module" },
      });
      const byTypeText =
        (byType.content as Array<{ type: string; text: string }>)[0]?.text ??
        "";
      expect(byTypeText).toContain("docs/okf/backend.md");
      expect(byTypeText).not.toContain("docs/okf/frontend.md");

      // tags filter round-trips.
      const byTags = await client.callTool({
        name: "oracle_search",
        arguments: { query: "doc", tags: ["okf", "backend"] },
      });
      const byTagsText =
        (byTags.content as Array<{ type: string; text: string }>)[0]?.text ??
        "";
      expect(byTagsText).toContain("docs/okf/backend.md");
      expect(byTagsText).not.toContain("docs/okf/frontend.md");

      // Tool schema also exposes the sources-expansion param.
      const searchProps = searchTool?.inputSchema.properties as Record<
        string,
        { type?: string }
      >;
      expect(searchProps).toHaveProperty("expand_sources");
      expect(searchProps.expand_sources.type).toBe("boolean");

      // Default (expand_sources omitted): the [expanded from ...] marker
      // appears for a seeded fmSources corpus. Mirrors mcp-server.test.ts's
      // "injects the [expanded from ...] marker ... by default" assertion.
      const withExpansion = await client.callTool({
        name: "oracle_search",
        arguments: { query: "module" },
      });
      const withExpansionText =
        (withExpansion.content as Array<{ type: string; text: string }>)[0]
          ?.text ?? "";
      expect(withExpansionText).toContain("docs/okf/module.md");
      expect(withExpansionText).toContain("src/impl.ts");
      expect(withExpansionText).toContain("[expanded from module.md]");
      expect(withExpansionText).toContain("impl body");
      const callsAfterExpansion = getFirstChunkByFile.mock.calls.length;
      expect(callsAfterExpansion).toBeGreaterThan(0);

      // expand_sources:false: no injection, no marker, and the resolver is
      // not consulted for THIS call (call count must not increase). Mirrors
      // mcp-server.test.ts's "respects expand_sources:false" assertion.
      const withoutExpansion = await client.callTool({
        name: "oracle_search",
        arguments: { query: "module", expand_sources: false },
      });
      const withoutExpansionText =
        (withoutExpansion.content as Array<{ type: string; text: string }>)[0]
          ?.text ?? "";
      expect(withoutExpansionText).toContain("docs/okf/module.md");
      expect(withoutExpansionText).not.toContain("[expanded from");
      expect(withoutExpansionText).not.toContain("impl body");
      expect(getFirstChunkByFile.mock.calls.length).toBe(callsAfterExpansion);
    } finally {
      try {
        await client.close();
      } catch {
        /* already closed */
      }
      try {
        await server.close();
      } catch {
        /* already closed */
      }
    }
  });
});

