#!/usr/bin/env node
import { fileURLToPath } from "node:url";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { loadEnvFromFile } from "./env.js";
import { loadConfig, type Config } from "./config.js";
import { createEmbeddings } from "./store/embeddings.js";
import { createVectorStore } from "./store/vector-store.js";
import { formatChunkLocation, queryCodebase, searchCodebase } from "./retrieval/chain.js";
import { formatRepoLine } from "./format-freshness.js";
import { expandFile, formatExpandResult } from "./expand.js";
import { formatIndexSummary, runIndex } from "./ingest/runner.js";
import { VERSION } from "./version.js";

loadEnvFromFile();

const config = loadConfig();

// ── Server factory ──────────────────────────────────────────────────────────
//
// Wrapping the server construction, per-instance state (storePromise /
// reindexInFlight / getStore), and tool registrations in a factory lets tests
// create an isolated instance without any real DB / network / stdio side
// effects, and lets startMcpServer() remain a thin wrapper that creates one
// production instance.
//
// Behavior is identical to the previous module-level wiring: in production
// startMcpServer() calls createMcpServer() once, so the per-instance
// storePromise / reindexInFlight state is functionally module-scoped.

export function createMcpServer(cfg: Config = config) {
  const server = new McpServer({
    name: "codebase-oracle",
    version: VERSION,
  });

  // Lazy-init store (expensive, only when first tool is called)
  let storePromise: ReturnType<typeof createVectorStore> | null = null;

  function getStore() {
    if (!storePromise) {
      const embeddings = createEmbeddings(cfg);
      storePromise = createVectorStore(embeddings, cfg).catch((err) => {
        // Don't cache the rejection: a config fix on disk should be visible to
        // the next tool call without restarting the server.
        storePromise = null;
        throw err;
      });
    }
    return storePromise;
  }

  // ── Tools ──────────────────────────────────────────────────────────────────

  server.tool(
    "oracle_query",
    "Ask a natural-language question about the indexed codebase. Returns an LLM-generated answer with source citations. Use this for understanding code, finding implementations, or learning how systems connect across repos.",
    {
      question: z.string().describe("Natural language question about the codebase"),
      repo: z.string().optional().describe("Optional: filter to a specific repo name (e.g. 'agent-tasks')"),
    },
    async ({ question, repo }) => {
      const store = await getStore();
      const result = await queryCodebase(question, store, cfg, { repo });

      const sourcesText = result.sources.length > 0
        ? "\n\nSources:\n" + result.sources.map((s) => `- ${s.filePath} (${s.repo})`).join("\n")
        : "";

      return { content: [{ type: "text" as const, text: result.answer + sourcesText }] };
    },
  );

  server.tool(
    "oracle_search",
    "Raw vector similarity search over the indexed codebase. Returns matching code/doc chunks with metadata. Use this when you need specific code snippets rather than an interpreted answer.",
    {
      query: z.string().describe("Search query (natural language or code pattern)"),
      repo: z.string().optional().describe("Optional: filter to a specific repo"),
      limit: z.number().int().min(1).max(50).optional().describe("Number of results (default 10)"),
      path_glob: z.string().optional().describe(
        "Optional glob on the chunk file path. picomatch semantics: `*` within a segment, `**` recursive, `?` single char, `{a,b}` alternatives. Example: `**/.github/workflows/*.yml`. AND-composes with `repo`. Note: the result count may fall short of `limit` for highly selective globs because the underlying over-fetch is capped to keep the SQLite scan bounded; raise `limit` if you need more matches.",
      ),
    },
    async ({ query, repo, limit, path_glob }) => {
      const store = await getStore();
      const docs = await searchCodebase(query, store, { repo, limit, pathGlob: path_glob });

      const text = docs
        .map((doc, i) => {
          const { repo: r } = doc.metadata as { repo: string };
          const location = formatChunkLocation(doc.metadata);
          return `[${i + 1}] ${location} (${r}):\n${doc.pageContent}`;
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

      const text = repos.map((r) => formatRepoLine(r)).join("\n");

      return {
        content: [{
          type: "text" as const,
          text: `${repos.length} indexed repos:\n${text}`,
        }],
      };
    },
  );

  server.tool(
    "oracle_expand",
    "Read a window of lines around a specific position in an indexed file. Use after oracle_search to see the context around a chunk without leaving the oracle. Reads the file from disk via the indexed absolutePath; if the working copy has changed since indexing, the lines may not match what oracle_search returned — check oracle_list_repos for the indexed timestamp.",
    {
      repo: z.string().describe("Repo name (must be indexed; see oracle_list_repos)"),
      path: z.string().describe("File path exactly as it appears in oracle_search results (e.g. `scaffoldkit/src/scaffoldkit/cli.py` — includes the repo segment)"),
      line: z.number().int().min(1).optional().describe("1-indexed line to center the window on (default 1, top of file)"),
      window: z.number().int().min(1).max(200).optional().describe("Lines to include around `line` (default 30, capped at 200)"),
    },
    async ({ repo, path, line, window }) => {
      const store = await getStore();
      const result = await expandFile(store, { repo, path, line, window });
      return { content: [{ type: "text" as const, text: formatExpandResult(result) }] };
    },
  );

  // Serialises oracle_reindex calls. Two parallel invocations would both
  // null out storePromise and then race their own SQLite write locks.
  // MCP stdio typically serialises in practice, but a one-line mutex turns
  // a possible BUSY error into a clear "already running" message.
  let reindexInFlight = false;

  server.tool(
    "oracle_reindex",
    "Run the indexing pipeline against the configured scan root. Incremental: only changed and new files are re-embedded, deleted files are pruned from the store. Use this after merging a PR you want the oracle to see immediately, instead of waiting for the next scheduled reindex.",
    {},
    async () => {
      if (reindexInFlight) {
        return {
          content: [{
            type: "text" as const,
            text: "Another oracle_reindex is already running. Wait for it to finish before triggering a new one.",
          }],
        };
      }
      reindexInFlight = true;
      try {
        // Drop the cached store handle so the indexer (which opens its own
        // SQLite connection) doesn't fight us for the write lock. Next
        // oracle_search / oracle_query call re-opens via getStore().
        if (storePromise) {
          try {
            const handle = await storePromise;
            handle.close();
          } catch {
            // Already closed or never opened cleanly; getStore() will handle it.
          }
          storePromise = null;
        }
        const summary = await runIndex(cfg);
        return { content: [{ type: "text" as const, text: formatIndexSummary(summary) }] };
      } finally {
        reindexInFlight = false;
      }
    },
  );

  return { server, getStore };
}

// ── Start ──────────────────────────────────────────────────────────────────

export async function startMcpServer(): Promise<void> {
  const { server } = createMcpServer();
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

// Only auto-start when invoked directly (e.g. `npm run mcp`, `node dist/mcp-server.js`).
// When imported by the CLI's `mcp` subcommand, the CLI calls startMcpServer() itself.
const invokedDirectly = process.argv[1]
  ? fileURLToPath(import.meta.url) === process.argv[1]
  : false;

if (invokedDirectly) {
  startMcpServer().catch((err) => {
    console.error("MCP server failed:", err);
    process.exit(1);
  });
}
