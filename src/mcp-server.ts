#!/usr/bin/env node
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { loadEnvFromFile } from "./env.js";
import { loadConfig } from "./config.js";
import { createEmbeddings } from "./store/embeddings.js";
import { createVectorStore } from "./store/vector-store.js";
import { queryCodebase, searchCodebase } from "./retrieval/chain.js";

loadEnvFromFile();

const config = loadConfig();

const server = new McpServer({
  name: "codebase-oracle",
  version: "0.2.0",
});

// Lazy-init store (expensive, only when first tool is called)
let storePromise: ReturnType<typeof createVectorStore> | null = null;

function getStore() {
  if (!storePromise) {
    const embeddings = createEmbeddings(config);
    storePromise = createVectorStore(embeddings, config);
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
    const result = await queryCodebase(question, store, config, { repo });

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
      content: [{
        type: "text" as const,
        text: `${repos.length} indexed repos:\n${text}`,
      }],
    };
  },
);

// ── Start ──────────────────────────────────────────────────────────────────

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch((err) => {
  console.error("MCP server failed:", err);
  process.exit(1);
});
