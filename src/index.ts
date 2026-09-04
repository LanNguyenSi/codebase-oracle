#!/usr/bin/env node
import { realpathSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { Command } from "commander";
import { loadEnvFromFile } from "./env.js";
import { loadConfig } from "./config.js";
import { createEmbeddings } from "./store/embeddings.js";
import {
  createVectorStore,
  IndexFingerprintError,
  listIndexedRepos,
} from "./store/vector-store.js";
import {
  formatChunkHeaderTags,
  formatChunkLocation,
  formatChunkSourcesLine,
  formatPointersSection,
  parseCommaSeparatedList,
  queryCodebase,
  searchCodebase,
} from "./retrieval/chain.js";
import { formatRepoLine } from "./format-freshness.js";
import { expandFile, formatExpandResult } from "./expand.js";
import { runWatchMode } from "./watch.js";
import { runMigrateStore } from "./migrate-store.js";
import { runIndex } from "./ingest/runner.js";
import { VERSION } from "./version.js";
import {
  formatErrorJson,
  formatExpandJson,
  formatQueryJson,
  formatReposJson,
  formatSearchJson,
} from "./format-json.js";

let jsonMode = false;
const JSON_COMMANDS = new Set(["query", "search", "list-repos", "expand"]);

export function isJsonCommandInvocation(args: string[]): boolean {
  const command = args[0];
  return command !== undefined && JSON_COMMANDS.has(command) && args.slice(1).includes("--json");
}

function jsonStoreLog(message: string): void {
  process.stderr.write(`${message}\n`);
}

// Builds the commander program without executing it, so tests can inspect
// the registered commands/options (e.g. to check the README stays in sync)
// without triggering a real CLI run.
export function buildProgram(): Command {
  const program = new Command();

  program
    .name("codebase-oracle")
    .description("RAG-powered codebase Q&A for your multi-repo codebase")
    .version(VERSION);

  program
    .command("mcp")
    .description("Start the Model Context Protocol server over stdio")
    .action(async () => {
      // Dynamic import so loadConfig() inside mcp-server doesn't run for
      // other subcommands that handle their own config loading.
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
        warn: (line) => process.stderr.write(line),
      });
    });

  program
    .command("query")
    .description("Ask a question about the codebase")
    .argument("<question>", "Natural language question")
    .option("-r, --repo <repo>", "Filter to a specific repo")
    .option("-k, --limit <limit>", "Number of chunks to retrieve", "12")
    .option("--json", "Output one JSON document")
    .action(async (question: string, opts) => {
      jsonMode = Boolean(opts.json);
      const config = loadConfig();
      const embeddings = createEmbeddings(config);
      const store = await createVectorStore(
        embeddings,
        config,
        undefined,
        opts.json ? jsonStoreLog : undefined,
      );

      try {
        const result = await queryCodebase(question, store, config, {
          repo: opts.repo,
          limit: parseInt(opts.limit, 10),
        });
        if (opts.json) {
          console.log(formatQueryJson(question, result));
          return;
        }
        console.log(`\nQuerying: "${question}"\n`);
        console.log(result.answer);
        if (result.sources.length > 0) {
          console.log("\n--- Sources ---");
          for (const source of result.sources) {
            console.log(`  ${source.filePath} (${source.repo})`);
          }
        }
        const pointersText = formatPointersSection(result.pointers);
        if (pointersText) {
          console.log(pointersText);
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
    .option(
      "-t, --type <type>",
      "Filter results by fmType chunk metadata (OKF frontmatter)",
    )
    .option(
      "--tags <tags>",
      "Filter results by fmTags chunk metadata (OKF frontmatter), comma-separated; ALL listed tags must match",
    )
    .option(
      "--no-expand-sources",
      "Disable OKF sources-expansion (do not inject files pointed at by a retrieved doc's `sources:` frontmatter)",
    )
    .option("--json", "Output one JSON document")
    .action(async (query: string, opts) => {
      jsonMode = Boolean(opts.json);
      const config = loadConfig();
      const embeddings = createEmbeddings(config);
      const store = await createVectorStore(
        embeddings,
        config,
        undefined,
        opts.json ? jsonStoreLog : undefined,
      );

      try {
        const docs = await searchCodebase(query, store, {
          repo: opts.repo,
          limit: parseInt(opts.limit, 10),
          pathGlob: opts.pathGlob,
          type: opts.type,
          tags: parseCommaSeparatedList(opts.tags),
          expandSources: opts.expandSources,
        });
        if (opts.json) {
          console.log(formatSearchJson(query, opts.repo, parseInt(opts.limit, 10), docs));
          return;
        }
        for (const doc of docs) {
          const { repo } = doc.metadata as { repo: string };
          const location = formatChunkLocation(doc.metadata);
          const tagSuffix = formatChunkHeaderTags(doc.metadata);
          console.log(`\n--- ${location} (${repo})${tagSuffix} ---`);
          const sourcesLine = formatChunkSourcesLine(doc.metadata);
          if (sourcesLine) console.log(sourcesLine);
          console.log(doc.pageContent.slice(0, 500));
        }
      } finally {
        store.close();
      }
    });

  program
    .command("list-repos")
    .description("List repos present in the vector index")
    .option("--json", "Output one JSON document")
    .action((opts) => {
      jsonMode = Boolean(opts.json);
      const config = loadConfig();
      const repos = listIndexedRepos(config);
      if (opts.json) {
        console.log(formatReposJson(repos));
        return;
      }
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
    .option(
      "-w, --window <n>",
      "Lines to include around the center (max 200)",
      "30",
    )
    .option("--json", "Output one JSON document")
    .action(async (repo, path, opts) => {
      jsonMode = Boolean(opts.json);
      const config = loadConfig();
      const embeddings = createEmbeddings(config);
      const store = await createVectorStore(
        embeddings,
        config,
        undefined,
        opts.json ? jsonStoreLog : undefined,
      );
      try {
        const result = await expandFile(store, {
          repo,
          path,
          line: parseInt(opts.line, 10),
          window: parseInt(opts.window, 10),
        });
        console.log(opts.json ? formatExpandJson(result) : formatExpandResult(result));
        if (!result.ok) process.exitCode = 1;
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
    .description(
      "Watch the scan root and re-embed files on change (debounced, incremental)",
    )
    .option("-p, --path <path>", "Path to scan root")
    .option(
      "--debounce <ms>",
      "Debounce window in ms after the last event",
      "3000",
    )
    .action(async (opts) => {
      const config = loadConfig(opts.path ? { scanRoot: opts.path } : {});
      const debounceMs = Number.parseInt(opts.debounce, 10);
      const watcher = await runWatchMode(config, {
        debounceMs:
          Number.isFinite(debounceMs) && debounceMs > 0
            ? debounceMs
            : undefined,
      });
      const shutdown = async (signal: NodeJS.Signals) => {
        console.log(`\nwatch: ${signal} received, flushing and closing...`);
        await watcher.close();
        process.exit(0);
      };
      process.on("SIGINT", () => void shutdown("SIGINT"));
      process.on("SIGTERM", () => void shutdown("SIGTERM"));
    });

  return program;
}

// process.argv[1] is path.resolve'd but keeps any symlink segment (e.g. the
// npm bin shim node_modules/.bin/codebase-oracle, or a global/npx install),
// while import.meta.url is realpath'd by Node, so a strict string compare
// silently misses "is this the entry module" for every symlinked bin. Resolve
// argv[1] through the filesystem too before comparing.
function resolveMainEntryPath(argv1: string | undefined): string | undefined {
  if (argv1 === undefined) return undefined;
  try {
    return realpathSync(argv1);
  } catch {
    return undefined;
  }
}

const isMainModule =
  resolveMainEntryPath(process.argv[1]) === fileURLToPath(import.meta.url);

if (isMainModule) {
  loadEnvFromFile();
  const program = buildProgram();
  jsonMode = isJsonCommandInvocation(process.argv.slice(2));
  if (jsonMode) {
    program.exitOverride();
    for (const command of program.commands) command.exitOverride();
  }
  Promise.resolve().then(() => program.parseAsync()).catch((err: unknown) => {
    if (jsonMode) {
      console.log(formatErrorJson(err));
      process.exitCode = 1;
      return;
    }
    if (err instanceof IndexFingerprintError) {
      console.error(err.message);
      process.exit(1);
    }
    throw err;
  });
}
