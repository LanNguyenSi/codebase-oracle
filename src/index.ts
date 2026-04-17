#!/usr/bin/env node
import { Command } from "commander";
import { loadEnvFromFile } from "./env.js";
import { loadConfig } from "./config.js";
import { discoverRepos, walkRepo } from "./ingest/scanner.js";
import { splitFile } from "./ingest/splitter.js";
import { createEmbeddings } from "./store/embeddings.js";
import {
  createVectorStore,
  IndexFingerprintError,
  listIndexedRepos,
} from "./store/vector-store.js";
import { openSqliteStore, type StoredEntry } from "./store/sqlite-store.js";
import { queryCodebase, searchCodebase } from "./retrieval/chain.js";
import { Document } from "@langchain/core/documents";
import { runWatchMode } from "./watch.js";
import { runMigrateStore } from "./migrate-store.js";

loadEnvFromFile();

const program = new Command();

function fileKey(repo: string, filePath: string): string {
  return `${repo}::${filePath}`;
}

program
  .name("codebase-oracle")
  .description("RAG-powered codebase Q&A for your multi-repo codebase")
  .version("0.3.0");

program
  .command("index")
  .description("Index all repos under the scan root")
  .option("-p, --path <path>", "Path to scan root")
  .action(async (opts) => {
    const config = loadConfig(opts.path ? { scanRoot: opts.path } : {});
    console.log(`Scanning repos in ${config.scanRoot}...`);

    const repos = await discoverRepos(config.scanRoot);
    console.log(`Found ${repos.length} repos`);

    const store = openSqliteStore(config);
    try {
      store.assertCompatibleWithConfig(config);
      const existingSignatures = store.fileSignatures();
      if (existingSignatures.size > 0) {
        console.log(
          `Loaded signatures for ${existingSignatures.size} files from ${store.dbPath} for incremental indexing`,
        );
      }

      const walkOptions = config.includeExtensions
        ? { extensions: new Set(config.includeExtensions) }
        : undefined;
      if (walkOptions) {
        console.log(
          `Using ORACLE_INCLUDE_EXTENSIONS override: ${config.includeExtensions!.join(", ")}`,
        );
      }

      let totalFiles = 0;
      let totalChunks = 0;
      let reusedFiles = 0;
      let changedFiles = 0;
      let newFiles = 0;
      let reusedChunks = 0;

      const seenKeys = new Set<string>();
      const docsToEmbed: Document[] = [];

      for (const repo of repos) {
        let repoFiles = 0;
        let repoChunks = 0;
        let repoReusedFiles = 0;
        process.stdout.write(`  ${repo.name}...`);

        for await (const file of walkRepo(repo.path, repo.name, config.scanRoot, walkOptions)) {
          repoFiles++;
          totalFiles++;

          const key = fileKey(file.repo, file.relativePath);
          seenKeys.add(key);
          const existing = existingSignatures.get(key);
          if (existing && existing.fileHash && existing.fileHash === file.contentHash) {
            reusedFiles++;
            repoReusedFiles++;
            continue;
          }

          if (existing) changedFiles++;
          else newFiles++;

          const chunks = await splitFile(file);
          docsToEmbed.push(...chunks);
          repoChunks += chunks.length;
          totalChunks += chunks.length;
        }

        console.log(` ${repoFiles} files, ${repoChunks} chunks (${repoReusedFiles} files reused)`);
      }

      // Files that existed in the store but were not seen this scan → deleted
      // on disk. Drop their vectors so stale chunks don't linger.
      let prunedFiles = 0;
      for (const [key, sig] of existingSignatures) {
        if (seenKeys.has(key)) continue;
        const removed = store.deleteByFile(sig.repo, sig.filePath);
        if (removed > 0) prunedFiles++;
      }
      if (prunedFiles > 0) {
        console.log(`Pruned ${prunedFiles} files that vanished from disk.`);
      }

      const filesToEmbed = changedFiles + newFiles;
      const countBeforeEmbed = store.count();

      console.log(
        `\nEmbedding ${docsToEmbed.length} chunks from ${filesToEmbed} changed/new files (${changedFiles} changed, ${newFiles} new). ${reusedFiles} files reused.`,
      );

      if (docsToEmbed.length === 0) {
        reusedChunks = countBeforeEmbed;
        totalChunks = countBeforeEmbed;
        console.log(
          `Index complete. ${totalFiles} files scanned, ${totalChunks} chunks total (${reusedChunks} reused, 0 newly embedded).`,
        );
        return;
      }

      // Initialize schema now that we know the embedding dimension (run the
      // first embed to discover it). If meta already exists, initializeSchema
      // is a no-op for matching inputs.
      const embeddings = createEmbeddings(config);
      const probeEmbedding = await embeddings.embedDocuments([docsToEmbed[0].pageContent]);
      if (probeEmbedding.length === 0 || probeEmbedding[0].length === 0) {
        throw new Error("Embedding provider returned empty vector for probe.");
      }
      store.initializeSchema({
        embeddingProvider: config.embeddingProvider,
        embeddingModel: config.embeddingModel,
        dimension: probeEmbedding[0].length,
      });

      // Group docs by file so upsertFile can atomically replace per-file chunks.
      const docsByFile = new Map<string, { repo: string; filePath: string; docs: Document[] }>();
      for (const doc of docsToEmbed) {
        const metadata = doc.metadata as { repo: string; filePath: string };
        const key = fileKey(metadata.repo, metadata.filePath);
        const group = docsByFile.get(key);
        if (group) {
          group.docs.push(doc);
        } else {
          docsByFile.set(key, {
            repo: metadata.repo,
            filePath: metadata.filePath,
            docs: [doc],
          });
        }
      }

      // Use the probe embedding for the first doc; batch-embed the rest in
      // chunks of 100.
      const firstDoc = docsToEmbed[0];
      const firstEntry: StoredEntry = {
        embedding: probeEmbedding[0],
        pageContent: firstDoc.pageContent,
        metadata: firstDoc.metadata as Record<string, unknown>,
      };
      const rest = docsToEmbed.slice(1);

      const embeddedByKey = new Map<string, StoredEntry[]>();
      const firstKey = fileKey(
        (firstDoc.metadata as { repo: string }).repo,
        (firstDoc.metadata as { filePath: string }).filePath,
      );
      embeddedByKey.set(firstKey, [firstEntry]);

      const batchSize = 100;
      for (let i = 0; i < rest.length; i += batchSize) {
        const batch = rest.slice(i, i + batchSize);
        const texts = batch.map((d) => d.pageContent);
        const embs = await embeddings.embedDocuments(texts);
        for (let j = 0; j < batch.length; j++) {
          const doc = batch[j];
          const metadata = doc.metadata as { repo: string; filePath: string };
          const key = fileKey(metadata.repo, metadata.filePath);
          const entry: StoredEntry = {
            embedding: embs[j],
            pageContent: doc.pageContent,
            metadata: doc.metadata as Record<string, unknown>,
          };
          const group = embeddedByKey.get(key);
          if (group) group.push(entry);
          else embeddedByKey.set(key, [entry]);
        }
        if (rest.length > batchSize) {
          process.stdout.write(
            `  Embedded ${Math.min(i + batchSize, rest.length) + 1}/${docsToEmbed.length}\r`,
          );
        }
      }
      if (rest.length > batchSize) console.log();

      // Transactionally upsert each file. upsertFile removes stale chunks for
      // that (repo, filePath) first, so changed files swap cleanly.
      for (const [key, entries] of embeddedByKey) {
        const group = docsByFile.get(key)!;
        const contentHash =
          (group.docs[0]?.metadata as { fileHash?: string })?.fileHash ?? null;
        store.upsertFile(group.repo, group.filePath, contentHash, entries);
      }

      const finalTotal = store.count();
      reusedChunks = finalTotal - docsToEmbed.length;
      console.log(
        `Index complete. ${totalFiles} files scanned, ${finalTotal} chunks total (${reusedChunks} reused, ${docsToEmbed.length} newly embedded).`,
      );
    } finally {
      store.close();
    }
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
  .action(async (query: string, opts) => {
    const config = loadConfig();
    const embeddings = createEmbeddings(config);
    const store = await createVectorStore(embeddings, config);

    try {
      const docs = await searchCodebase(query, store, {
        repo: opts.repo,
        limit: parseInt(opts.limit, 10),
      });
      for (const doc of docs) {
        const { repo, filePath } = doc.metadata as { repo: string; filePath: string };
        console.log(`\n--- ${filePath} (${repo}) ---`);
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
      console.log(`  ${r.repo} — ${r.chunkCount} chunks across ${r.fileCount} files`);
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
