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
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { createHttpRequestHandler } from "../../src/http-server.js";
import type { HttpBindConfig } from "../../src/http-auth.js";
import { VERSION } from "../../src/version.js";

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
        close: () => new Promise<void>((res, rej) => server.close((err) => (err ? rej(err) : res()))),
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
    const body = await res.json() as { jsonrpc: string; error: { code: number; message: string }; id: null };
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
    const body = await res.json() as { error: { code: number; message: string } };
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
      body: JSON.stringify({ jsonrpc: "2.0", id: 1, method: "initialize", params: { protocolVersion: "2024-11-05", capabilities: {}, clientInfo: { name: "test", version: "1.0" } } }),
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
    const body = await res.json() as { status: string; version: string };
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
    const body = await res.json() as { error: string };
    expect(body.error).toContain("Not found");
  });

  it("POST /unknown → 404", async () => {
    const res = await fetch(`${url}/unknown`, { method: "POST" });
    expect(res.status).toBe(404);
  });
});
