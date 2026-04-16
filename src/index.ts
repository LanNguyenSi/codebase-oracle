#!/usr/bin/env node
import { Command } from "commander";
import { loadConfig } from "./config.js";
import { discoverRepos, walkRepo } from "./ingest/scanner.js";
import { splitFile } from "./ingest/splitter.js";
import { createEmbeddings } from "./store/embeddings.js";
import { createVectorStore, persistIndex } from "./store/vector-store.js";
import { queryCodebase, searchCodebase } from "./retrieval/chain.js";
import type { Document } from "@langchain/core/documents";

const program = new Command();

program
  .name("codebase-oracle")
  .description("RAG-powered codebase Q&A for the your multi-repo codebase")
  .version("0.1.0");

program
  .command("index")
  .description("Index all repos under the scan root")
  .option("-p, --path <path>", "Path to scan root")
  .action(async (opts) => {
    const config = loadConfig({ scanRoot: opts.path });
    console.log(`Scanning repos in ${config.scanRoot}...`);

    const repos = await discoverRepos(config.scanRoot);
    console.log(`Found ${repos.length} repos`);

    const embeddings = createEmbeddings(config);
    const allDocs: Document[] = [];
    let totalFiles = 0;

    for (const repo of repos) {
      let repoFiles = 0;
      process.stdout.write(`  ${repo.name}...`);

      for await (const file of walkRepo(repo.path, repo.name, config.scanRoot)) {
        const chunks = await splitFile(file);
        allDocs.push(...chunks);
        repoFiles++;
      }

      console.log(` ${repoFiles} files, ${allDocs.length} chunks total`);
      totalFiles += repoFiles;
    }

    console.log(`\nEmbedding ${allDocs.length} chunks from ${totalFiles} files...`);

    // Persist raw documents for re-embedding without re-scanning
    await persistIndex(allDocs, config);

    // Build vector store (this embeds all documents)
    const store = await createVectorStore(embeddings, config);
    await store.addDocuments(allDocs);

    console.log("Index complete.");
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

    const docs = await searchCodebase(query, store, {
      repo: opts.repo,
      limit: parseInt(opts.limit, 10),
    });

    for (const doc of docs) {
      const { repo, filePath } = doc.metadata as { repo: string; filePath: string };
      console.log(`\n--- ${filePath} (${repo}) ---`);
      console.log(doc.pageContent.slice(0, 500));
    }
  });

program.parse();
