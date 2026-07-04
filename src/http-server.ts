#!/usr/bin/env node
/**
 * HTTP-based MCP server for codebase-oracle.
 *
 * Exposes four of the five stdio MCP tools (oracle_query, oracle_search,
 * oracle_list_repos, oracle_expand) over Streamable HTTP so that local
 * agents can connect without stdio. oracle_reindex and oracle_search's
 * path_glob filter are intentionally stdio-only and not registered here.
 *
 * Usage:
 *   npm run serve                         # default port 3100
 *   ORACLE_HTTP_PORT=8080 npm run serve   # custom port
 *
 * Connect from Claude Code:
 *   claude mcp add codebase-oracle --transport http http://localhost:3100/mcp
 */
import { fileURLToPath } from "node:url";
import { createServer as createHttpServer } from "node:http";
import type { IncomingMessage, ServerResponse } from "node:http";
import { Readable } from "node:stream";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { WebStandardStreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/webStandardStreamableHttp.js";
import { z } from "zod";
import { loadEnvFromFile } from "./env.js";
import { loadConfig } from "./config.js";
import { createEmbeddings } from "./store/embeddings.js";
import {
  createVectorStore,
  IndexFingerprintError,
  type VectorStoreWrapper,
} from "./store/vector-store.js";
import {
  formatPointersSection,
  formatSearchResults,
  queryCodebase,
  searchCodebase,
} from "./retrieval/chain.js";
import { formatRepoLine } from "./format-freshness.js";
import { expandFile, formatExpandResult } from "./expand.js";
import {
  resolveHttpBindConfig,
  verifyBearer,
  type HttpBindConfig,
} from "./http-auth.js";
import { VERSION } from "./version.js";

loadEnvFromFile();

const config = loadConfig();

// ── Lazy vector store ─────────────────────────────────────────────────────────

let storePromise: Promise<VectorStoreWrapper> | null = null;

function getStore(): Promise<VectorStoreWrapper> {
  if (!storePromise) {
    const embeddings = createEmbeddings(config);
    storePromise = createVectorStore(embeddings, config).catch((err) => {
      // Don't cache the rejection: a config fix on disk should be visible to
      // the next tool call without restarting the server.
      storePromise = null;
      throw err;
    });
  }
  return storePromise;
}

// ── MCP server factory ────────────────────────────────────────────────────────

// One McpServer cannot be connected to more than one transport at a time: the
// SDK's Protocol.connect throws "Already connected to a transport" once
// server._transport is set, and a stateless POST never clears it. So we build a
// fresh McpServer per request. Tools close over the shared lazy getStore()/config,
// so there is no per-request store rebuild.
// Exported so tests can connect an in-memory MCP client directly to a fresh
// server instance without going through the HTTP transport (mirrors
// mcp-server.ts's createMcpServer() test seam).
export function buildServer(): McpServer {
  const server = new McpServer({
    name: "codebase-oracle",
    version: VERSION,
  });

  server.tool(
    "oracle_query",
    "Ask a natural-language question about the indexed codebase. Returns an LLM-generated answer with source citations. Use this for understanding code, finding implementations, or learning how systems connect across repos.",
    {
      question: z
        .string()
        .describe("Natural language question about the codebase"),
      repo: z
        .string()
        .optional()
        .describe(
          "Optional: filter to a specific repo name (e.g. 'agent-tasks')",
        ),
    },
    async ({ question, repo }) => {
      const store = await getStore();
      const result = await queryCodebase(question, store, config, { repo });

      const sourcesText =
        result.sources.length > 0
          ? "\n\nSources:\n" +
            result.sources.map((s) => `- ${s.filePath} (${s.repo})`).join("\n")
          : "";
      const pointersText = formatPointersSection(result.pointers);

      return {
        content: [
          {
            type: "text" as const,
            text: result.answer + sourcesText + pointersText,
          },
        ],
      };
    },
  );

  server.tool(
    "oracle_search",
    "Raw vector similarity search over the indexed codebase. Returns matching code/doc chunks with metadata. Use this when you need specific code snippets rather than an interpreted answer. No LLM involved — pure embedding retrieval.",
    {
      query: z
        .string()
        .describe("Search query (natural language or code pattern)"),
      repo: z
        .string()
        .optional()
        .describe("Optional: filter to a specific repo"),
      limit: z
        .number()
        .int()
        .min(1)
        .max(50)
        .optional()
        .describe("Number of results (default 10)"),
      type: z
        .string()
        .optional()
        .describe(
          "Optional: filter to chunks whose fmType OKF frontmatter metadata strictly equals this value. AND-composes with `repo`/`tags`. Chunks without fmType (no frontmatter, or frontmatter missing a `type` field) are excluded when this is set. Note: the result count may fall short of `limit` for highly selective filters because the underlying over-fetch is capped to keep the SQLite scan bounded; raise `limit` if you need more matches.",
        ),
      tags: z
        .array(z.string())
        .optional()
        .describe(
          "Optional: filter to chunks whose fmTags OKF frontmatter metadata contains ALL of the listed tags. AND-composes with `repo`/`type`. Chunks without fmTags (no frontmatter, or frontmatter missing a `tags` field) are excluded when this is set. Note: the result count may fall short of `limit` for highly selective filters because the underlying over-fetch is capped to keep the SQLite scan bounded; raise `limit` if you need more matches.",
        ),
      expand_sources: z
        .boolean()
        .optional()
        .describe(
          "inject files pointed at by a retrieved doc's OKF sources: frontmatter, marked [expanded from ...] (default true)",
        ),
    },
    async ({ query, repo, limit, type, tags, expand_sources }) => {
      const store = await getStore();
      const docs = await searchCodebase(query, store, {
        repo,
        limit,
        type,
        tags,
        expandSources: expand_sources,
      });
      return {
        content: [{ type: "text" as const, text: formatSearchResults(docs) }],
      };
    },
  );

  server.tool(
    "oracle_list_repos",
    "List repos actually present in the vector index, with chunk and file counts. Reflects what oracle_search / oracle_query can answer over — not just what exists on disk.",
    {},
    async () => {
      const store = await getStore();
      const repos = store.listRepos();

      if (repos.length === 0) {
        return {
          content: [
            {
              type: "text" as const,
              text: "No repos in the index yet. Run `npm run index` to build it.",
            },
          ],
        };
      }

      const text = repos.map((r) => formatRepoLine(r)).join("\n");

      return {
        content: [
          {
            type: "text" as const,
            text: `${repos.length} indexed repos:\n${text}`,
          },
        ],
      };
    },
  );

  server.tool(
    "oracle_expand",
    "Read a window of lines around a specific position in an indexed file. Use after oracle_search to see the context around a chunk without leaving the oracle. Reads the file from disk via the indexed absolutePath; if the working copy has changed since indexing, the lines may not match what oracle_search returned — check oracle_list_repos for the indexed timestamp.",
    {
      repo: z
        .string()
        .describe("Repo name (must be indexed; see oracle_list_repos)"),
      path: z
        .string()
        .describe(
          "File path exactly as it appears in oracle_search results (e.g. `scaffoldkit/src/scaffoldkit/cli.py` — includes the repo segment)",
        ),
      line: z
        .number()
        .int()
        .min(1)
        .optional()
        .describe(
          "1-indexed line to center the window on (default 1, top of file)",
        ),
      window: z
        .number()
        .int()
        .min(1)
        .max(200)
        .optional()
        .describe("Lines to include around `line` (default 30, capped at 200)"),
    },
    async ({ repo, path, line, window }) => {
      const store = await getStore();
      const result = await expandFile(store, { repo, path, line, window });
      return {
        content: [{ type: "text" as const, text: formatExpandResult(result) }],
      };
    },
  );

  return server;
}

// ── HTTP request handler factory ──────────────────────────────────────────────
//
// Extracting the handler into a factory makes the auth gate and routing
// testable without binding a real port at import time. startHttpServer()
// wires the factory into a node:http.Server and calls .listen(); tests call
// createHttpRequestHandler() directly and pass the result to http.createServer,
// then listen on port 0 for an ephemeral port.
//
// The handler body is byte-for-byte identical to the original inline handler.

export function createHttpRequestHandler(
  bindConfig: HttpBindConfig,
  port: number = 3100,
): (req: IncomingMessage, res: ServerResponse) => Promise<void> {
  return async (req, res) => {
    const url = new URL(req.url ?? "/", `http://localhost:${port}`);

    // Health check
    if (req.method === "GET" && url.pathname === "/health") {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ status: "ok", version: VERSION }));
      return;
    }

    // MCP endpoint
    if (req.method === "POST" && url.pathname === "/mcp") {
      const auth = verifyBearer(req.headers.authorization, bindConfig.token);
      if (!auth.ok) {
        res.writeHead(401, {
          "Content-Type": "application/json",
          "WWW-Authenticate": 'Bearer realm="codebase-oracle"',
        });
        res.end(
          JSON.stringify({
            jsonrpc: "2.0",
            error: {
              code: -32001,
              message: `Unauthorized: ${auth.reason} bearer token`,
            },
            id: null,
          }),
        );
        return;
      }

      try {
        // Collect request body
        const chunks: Buffer[] = [];
        for await (const chunk of req) chunks.push(chunk as Buffer);
        const body = Buffer.concat(chunks);

        // Build standard Request object for the MCP transport
        const headers = new Headers();
        for (const [key, value] of Object.entries(req.headers)) {
          if (value)
            headers.set(key, Array.isArray(value) ? value.join(", ") : value);
        }

        const request = new Request(url.toString(), {
          method: "POST",
          headers,
          body,
        });

        // Stateless: a fresh server + transport per request. Reusing one McpServer
        // across requests fails on the second POST ("Already connected to a
        // transport") because the SDK never clears server._transport on a normal
        // stateless POST.
        const server = buildServer();
        const transport = new WebStandardStreamableHTTPServerTransport({
          sessionIdGenerator: undefined,
        });

        await server.connect(transport);
        const response = await transport.handleRequest(request);

        // Release the per-request server/transport once the body has been fully
        // streamed (or immediately when there is no body), so we don't leak a
        // connected transport per request.
        const releasePair = () => {
          void transport.close().catch(() => {});
          void server.close().catch(() => {});
        };

        // Forward response headers
        res.writeHead(
          response.status,
          Object.fromEntries(response.headers.entries()),
        );

        // Stream the response body (supports SSE)
        if (response.body) {
          const nodeStream = Readable.fromWeb(response.body as any);
          nodeStream.on("close", releasePair);
          nodeStream.on("error", releasePair);
          nodeStream.pipe(res);
        } else {
          releasePair();
          res.end();
        }
      } catch (err) {
        console.error("[http-mcp] Error:", err);
        if (!res.headersSent) {
          res.writeHead(500, { "Content-Type": "application/json" });
        }
        const message =
          err instanceof IndexFingerprintError ? err.message : "Internal error";
        res.end(
          JSON.stringify({
            jsonrpc: "2.0",
            error: { code: -32603, message },
            id: null,
          }),
        );
      }
      return;
    }

    // 404 for everything else
    res.writeHead(404, { "Content-Type": "application/json" });
    res.end(
      JSON.stringify({
        error: "Not found. POST /mcp for MCP, GET /health for status.",
      }),
    );
  };
}

// ── HTTP server startup ───────────────────────────────────────────────────────

export function startHttpServer(): void {
  let bindConfig: HttpBindConfig;
  try {
    bindConfig = resolveHttpBindConfig(process.env);
  } catch (err) {
    console.error(err instanceof Error ? err.message : err);
    process.exit(1);
  }
  const port = Number(process.env.ORACLE_HTTP_PORT ?? 3100);
  const server = createHttpServer(createHttpRequestHandler(bindConfig, port));
  server.listen(port, bindConfig.bind, () => {
    const authNote = bindConfig.token
      ? "with bearer-token auth"
      : "(no auth; loopback only)";
    console.log(
      `[codebase-oracle] HTTP MCP server listening on http://${bindConfig.bind}:${port}/mcp ${authNote}`,
    );
    console.log(
      `[codebase-oracle] Health check: http://${bindConfig.bind}:${port}/health`,
    );
  });
}

// Only auto-start when invoked directly (e.g. `npm run serve`, `node dist/http-server.js`).
// Importing this module (e.g. for tests) does NOT bind a port or call process.exit.
//
// Before this guard was added, the module would:
//   process.exit(1) when resolveHttpBindConfig(process.env) threw (no-token off-loopback bind)
//   httpServer.listen(port, ...) unconditionally at module load time.
// Now those side effects live exclusively inside startHttpServer(), which only
// runs when the file is invoked directly.
const invokedDirectly = process.argv[1]
  ? fileURLToPath(import.meta.url) === process.argv[1]
  : false;

if (invokedDirectly) startHttpServer();
