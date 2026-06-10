#!/usr/bin/env node
import { Command } from "commander";
import { loadEnvFromFile } from "./env.js";
import { loadConfig } from "./config.js";
import { createEmbeddings } from "./store/embeddings.js";
import {
  createVectorStore,
  IndexFingerprintError,
  listIndexedRepos,
} from "./store/vector-store.js";
import { formatChunkLocation, queryCodebase, searchCodebase } from "./retrieval/chain.js";
import { formatRepoLine } from "./format-freshness.js";
import { expandFile, formatExpandResult } from "./expand.js";
import { runWatchMode } from "./watch.js";
import { runMigrateStore } from "./migrate-store.js";
import { runIndex } from "./ingest/runner.js";
import { VERSION } from "./version.js";

loadEnvFromFile();

const program = new Command();

program
  .name("codebase-oracle")
  .description("RAG-powered codebase Q&A for your multi-repo codebase")
  .version(VERSION);

program
  .command("mcp")
  .description("Start the Model Context Protocol server over stdio")
  .action(async () => {
    // Dynamic import so loadConfig() inside mcp-server doesn't run for other
    // subcommands that handle their own config loading.
    const { startMcpServer } = await import("./mcp-server.js");
    await startMcpServer();
  });

program
  .command("index")
  .description("Index all repos under the scan root")
  .option("-p, --path <path>", "Path to scan root")
  .action(async (opts) => {
    const config = loadConfig(opts.path ? { scanRoot: opts.path } : {});
    await runIndex(config, {
      logger: (line) => process.stdout.write(line),
    });
  });

program
  .command("query")
  .description("Ask a question about the codebase")
  .argument("<question>", "Natural language question")
  .option("-r, --repo <repo>", "Filter to a specific repo")
  .option("-k, --limit <limit>", "Number of chunks to retrieve", "12")
  .action(async (question: string, opts) => {
    const config = loadConfig();
    const embeddings = createEmbeddings(config);
    const store = await createVectorStore(embeddings, config);

    try {
      console.log(`\nQuerying: "${question}"\n`);
      const result = await queryCodebase(question, store, config, {
        repo: opts.repo,
        limit: parseInt(opts.limit, 10),
      });
      console.log(result.answer);
      if (result.sources.length > 0) {
        console.log("\n--- Sources ---");
        for (const source of result.sources) {
          console.log(`  ${source.filePath} (${source.repo})`);
        }
      }
    } finally {
      store.close();
    }
  });

program
  .command("search")
  .description("Raw vector search (returns matching chunks)")
  .argument("<query>", "Search query")
  .option("-r, --repo <repo>", "Filter to a specific repo")
  .option("-k, --limit <limit>", "Number of results", "10")
  .option(
    "-g, --path-glob <glob>",
    "Filter results by file path glob (e.g. **/.github/workflows/*.yml)",
  )
  .action(async (query: string, opts) => {
    const config = loadConfig();
    const embeddings = createEmbeddings(config);
    const store = await createVectorStore(embeddings, config);

    try {
      const docs = await searchCodebase(query, store, {
        repo: opts.repo,
        limit: parseInt(opts.limit, 10),
        pathGlob: opts.pathGlob,
      });
      for (const doc of docs) {
        const { repo } = doc.metadata as { repo: string };
        const location = formatChunkLocation(doc.metadata);
        console.log(`\n--- ${location} (${repo}) ---`);
        console.log(doc.pageContent.slice(0, 500));
      }
    } finally {
      store.close();
    }
  });

program
  .command("list-repos")
  .description("List repos present in the vector index")
  .action(() => {
    const config = loadConfig();
    const repos = listIndexedRepos(config);
    if (repos.length === 0) {
      console.log("No repos indexed yet. Run `npm run index`.");
      return;
    }
    for (const r of repos) {
      console.log(formatRepoLine(r, { prefix: "  " }));
    }
  });

program
  .command("expand")
  .description("Read a window of lines around a position in an indexed file")
  .argument("<repo>", "Repo name (must be indexed)")
  .argument("<path>", "File path relative to the repo root")
  .option("-l, --line <n>", "1-indexed line to center the window on", "1")
  .option("-w, --window <n>", "Lines to include around the center (max 200)", "30")
  .action(async (repo, path, opts) => {
    const config = loadConfig();
    const embeddings = createEmbeddings(config);
    const store = await createVectorStore(embeddings, config);
    try {
      const result = await expandFile(store, {
        repo,
        path,
        line: parseInt(opts.line, 10),
        window: parseInt(opts.window, 10),
      });
      console.log(formatExpandResult(result));
      if (!result.ok) process.exit(1);
    } finally {
      store.close();
    }
  });

program
  .command("migrate-store")
  .description("Migrate a v0.2.0 embeddings.jsonl to the v0.3.0 SQLite store")
  .action(async () => {
    const config = loadConfig();
    await runMigrateStore(config);
  });

program
  .command("watch")
  .description("Watch the scan root and re-embed files on change (debounced, incremental)")
  .option("-p, --path <path>", "Path to scan root")
  .option("--debounce <ms>", "Debounce window in ms after the last event", "3000")
  .action(async (opts) => {
    const config = loadConfig(opts.path ? { scanRoot: opts.path } : {});
    const debounceMs = Number.parseInt(opts.debounce, 10);
    const watcher = await runWatchMode(config, {
      debounceMs: Number.isFinite(debounceMs) && debounceMs > 0 ? debounceMs : undefined,
    });
    const shutdown = async (signal: NodeJS.Signals) => {
      console.log(`\nwatch: ${signal} received, flushing and closing...`);
      await watcher.close();
      process.exit(0);
    };
    process.on("SIGINT", () => void shutdown("SIGINT"));
    process.on("SIGTERM", () => void shutdown("SIGTERM"));
  });

program.parseAsync().catch((err: unknown) => {
  if (err instanceof IndexFingerprintError) {
    console.error(err.message);
    process.exit(1);
  }
  throw err;
});
