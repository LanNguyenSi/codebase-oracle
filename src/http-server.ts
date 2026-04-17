#!/usr/bin/env node
/**
 * HTTP-based MCP server for codebase-oracle.
 *
 * Exposes the same 3 tools as the stdio MCP server (oracle_query,
 * oracle_search, oracle_list_repos) over Streamable HTTP so that
 * local agents can connect without stdio.
 *
 * Usage:
 *   npm run serve                         # default port 3100
 *   ORACLE_HTTP_PORT=8080 npm run serve   # custom port
 *
 * Connect from Claude Code:
 *   claude mcp add codebase-oracle --transport http http://localhost:3100/mcp
 */
import { createServer as createHttpServer } from "node:http";
import { Readable } from "node:stream";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { WebStandardStreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/webStandardStreamableHttp.js";
import { z } from "zod";
import { loadEnvFromFile } from "./env.js";
import { loadConfig } from "./config.js";
import { createEmbeddings } from "./store/embeddings.js";
import { createVectorStore, IndexFingerprintError, type VectorStoreWrapper } from "./store/vector-store.js";
import { queryCodebase, searchCodebase } from "./retrieval/chain.js";
import { resolveHttpBindConfig, verifyBearer } from "./http-auth.js";

loadEnvFromFile();

let bindConfig;
try {
  bindConfig = resolveHttpBindConfig(process.env);
} catch (err) {
  console.error(err instanceof Error ? err.message : err);
  process.exit(1);
}

const config = loadConfig();
const port = Number(process.env.ORACLE_HTTP_PORT ?? 3100);

// ── Lazy vector store ─────────────────────────────────────────────────────────

let storePromise: Promise<VectorStoreWrapper> | null = null;

function getStore(): Promise<VectorStoreWrapper> {
  if (!storePromise) {
    const embeddings = createEmbeddings(config);
    storePromise = createVectorStore(embeddings, config);
  }
  return storePromise;
}

// ── MCP server (singleton) ────────────────────────────────────────────────────

const server = new McpServer({
  name: "codebase-oracle",
  version: "0.1.0",
});

server.tool(
  "oracle_query",
  "Ask a natural-language question about the indexed codebase. Returns an LLM-generated answer with source citations. Use this for understanding code, finding implementations, or learning how systems connect across repos.",
  {
    question: z.string().describe("Natural language question about the codebase"),
    repo: z.string().optional().describe("Optional: filter to a specific repo name (e.g. 'agent-tasks')"),
  },
  async ({ question, repo }) => {
    const store = await getStore();
    const result = await queryCodebase(question, store, config, { repo });

    const sourcesText = result.sources.length > 0
      ? "\n\nSources:\n" + result.sources.map((s) => `- ${s.filePath} (${s.repo})`).join("\n")
      : "";

    return { content: [{ type: "text" as const, text: result.answer + sourcesText }] };
  },
);

server.tool(
  "oracle_search",
  "Raw vector similarity search over the indexed codebase. Returns matching code/doc chunks with metadata. Use this when you need specific code snippets rather than an interpreted answer. No LLM involved — pure embedding retrieval.",
  {
    query: z.string().describe("Search query (natural language or code pattern)"),
    repo: z.string().optional().describe("Optional: filter to a specific repo"),
    limit: z.number().int().min(1).max(50).optional().describe("Number of results (default 10)"),
  },
  async ({ query, repo, limit }) => {
    const store = await getStore();
    const docs = await searchCodebase(query, store, { repo, limit });

    const text = docs
      .map((doc, i) => {
        const { repo: r, filePath } = doc.metadata as { repo: string; filePath: string };
        return `[${i + 1}] ${filePath} (${r}):\n${doc.pageContent}`;
      })
      .join("\n\n---\n\n");

    return { content: [{ type: "text" as const, text: text || "No results found." }] };
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
        content: [{
          type: "text" as const,
          text: "No repos in the index yet. Run `npm run index` to build it.",
        }],
      };
    }

    const text = repos
      .map((r) => `- ${r.repo} — ${r.chunkCount} chunks across ${r.fileCount} files`)
      .join("\n");

    return {
      content: [{ type: "text" as const, text: `${repos.length} indexed repos:\n${text}` }],
    };
  },
);

// ── Node HTTP server ──────────────────────────────────────────────────────────

const httpServer = createHttpServer(async (req, res) => {
  const url = new URL(req.url ?? "/", `http://localhost:${port}`);

  // Health check
  if (req.method === "GET" && url.pathname === "/health") {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ status: "ok", version: "0.1.0" }));
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
          error: { code: -32001, message: `Unauthorized: ${auth.reason} bearer token` },
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
        if (value) headers.set(key, Array.isArray(value) ? value.join(", ") : value);
      }

      const request = new Request(url.toString(), {
        method: "POST",
        headers,
        body,
      });

      // Stateless transport — one per request, server is shared
      const transport = new WebStandardStreamableHTTPServerTransport({
        sessionIdGenerator: undefined,
      });

      await server.connect(transport);
      const response = await transport.handleRequest(request);

      // Forward response headers
      res.writeHead(response.status, Object.fromEntries(response.headers.entries()));

      // Stream the response body (supports SSE)
      if (response.body) {
        const nodeStream = Readable.fromWeb(response.body as any);
        nodeStream.pipe(res);
      } else {
        res.end();
      }
    } catch (err) {
      console.error("[http-mcp] Error:", err);
      if (!res.headersSent) {
        res.writeHead(500, { "Content-Type": "application/json" });
      }
      const message = err instanceof IndexFingerprintError
        ? err.message
        : "Internal error";
      res.end(JSON.stringify({ jsonrpc: "2.0", error: { code: -32603, message }, id: null }));
    }
    return;
  }

  // 404 for everything else
  res.writeHead(404, { "Content-Type": "application/json" });
  res.end(JSON.stringify({ error: "Not found. POST /mcp for MCP, GET /health for status." }));
});

httpServer.listen(port, bindConfig.bind, () => {
  const authNote = bindConfig.token ? "with bearer-token auth" : "(no auth; loopback only)";
  console.log(`[codebase-oracle] HTTP MCP server listening on http://${bindConfig.bind}:${port}/mcp ${authNote}`);
  console.log(`[codebase-oracle] Health check: http://${bindConfig.bind}:${port}/health`);
});
