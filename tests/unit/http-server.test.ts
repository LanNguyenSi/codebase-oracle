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
 * GET /health → 200, unknown path → 404.
 *
 * The IndexFingerprintError → 500 path requires mocking the store or the
 * MCP tool layer; it is flagged as a risk/open question in the task report.
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
import { createVectorStore } from "../../src/store/vector-store.js";
import type { VectorStoreWrapper } from "../../src/store/vector-store.js";

// Undo the vi.stubEnv from the vi.hoisted block above once every test in
// this file has run, so the stubbed ORACLE_EMBEDDING_PROVIDER never leaks
// into a different test file sharing the same worker process.
afterAll(() => {
  vi.unstubAllEnvs();
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

// ── oracle_search type/tags pass-through (OKF metadata) ──────────────────────
//
// Connects an MCP client directly to buildServer() via InMemoryTransport,
// bypassing the HTTP transport entirely (mirrors mcp-server.test.ts's
// connectClient() pattern). createVectorStore is mocked so no real DB or
// embeddings are touched. Kept to a single test with multiple calls sharing
// one connection: http-server.ts's getStore()/storePromise cache is
// module-level (not per-instance like mcp-server.ts's createMcpServer()), so
// splitting this across multiple `it` blocks would let a later test observe
// the earlier test's cached store instead of its own mock.
describe("oracle_search type/tags pass-through (HTTP MCP)", () => {
  function fakeStore(
    similaritySearch: VectorStoreWrapper["similaritySearch"],
  ): VectorStoreWrapper {
    return {
      addDocuments: async () => {},
      similaritySearch,
      listRepos: () => [],
      getFileMetadata: () => null,
      close: () => {},
    };
  }

  it("passes `type` and `tags` through to the real filtering logic", async () => {
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
    vi.mocked(createVectorStore).mockResolvedValue(fakeStore(async () => docs));

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
