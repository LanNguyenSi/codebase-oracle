#!/usr/bin/env node
import { Command } from "commander";
import { loadEnvFromFile } from "./env.js";
import { loadConfig } from "./config.js";
import { discoverRepos, walkRepo } from "./ingest/scanner.js";
import { splitFile } from "./ingest/splitter.js";
import { createEmbeddings } from "./store/embeddings.js";
import {
  appendStoredVectors,
  createVectorStore,
  initializeStoredVectors,
  loadStoredVectors,
  persistIndex,
  type StoredVector,
} from "./store/vector-store.js";
import { queryCodebase, searchCodebase } from "./retrieval/chain.js";
import { Document } from "@langchain/core/documents";

loadEnvFromFile();

const program = new Command();

function fileKey(repo: string, filePath: string): string {
  return `${repo}::${filePath}`;
}

function groupVectorsByFile(vectors: StoredVector[]): Map<string, { vectors: StoredVector[]; fileHash: string | null }> {
  const groups = new Map<string, { vectors: StoredVector[]; fileHash: string | null }>();

  for (const vector of vectors) {
    const metadata = vector.doc.metadata;
    const repo = typeof metadata.repo === "string" ? metadata.repo : null;
    const filePath = typeof metadata.filePath === "string" ? metadata.filePath : null;
    if (!repo || !filePath) continue;

    const hash = typeof metadata.fileHash === "string" ? metadata.fileHash : null;
    const key = fileKey(repo, filePath);
    const group = groups.get(key);

    if (!group) {
      groups.set(key, { vectors: [vector], fileHash: hash });
      continue;
    }

    group.vectors.push(vector);
    if (!group.fileHash || !hash || group.fileHash !== hash) {
      group.fileHash = null;
    }
  }

  return groups;
}

program
  .name("codebase-oracle")
  .description("RAG-powered codebase Q&A for the your multi-repo codebase")
  .version("0.1.0");

program
  .command("index")
  .description("Index all repos under the scan root")
  .option("-p, --path <path>", "Path to scan root")
  .action(async (opts) => {
    const config = loadConfig(opts.path ? { scanRoot: opts.path } : {});
    console.log(`Scanning repos in ${config.scanRoot}...`);

    const repos = await discoverRepos(config.scanRoot);
    console.log(`Found ${repos.length} repos`);

    const persistedVectors = await loadStoredVectors(config);
    const persistedByFile = groupVectorsByFile(persistedVectors);
    if (persistedVectors.length > 0) {
      console.log(`Loaded ${persistedVectors.length} existing vectors for incremental indexing`);
    }

    const reusedVectors: StoredVector[] = [];
    const docsToEmbed: Document[] = [];

    let totalFiles = 0;
    let totalChunks = 0;
    let reusedFiles = 0;
    let changedFiles = 0;
    let newFiles = 0;
    let reusedChunks = 0;

    for (const repo of repos) {
      let repoFiles = 0;
      let repoChunks = 0;
      let repoReusedFiles = 0;
      process.stdout.write(`  ${repo.name}...`);

      for await (const file of walkRepo(repo.path, repo.name, config.scanRoot)) {
        repoFiles++;
        totalFiles++;

        const key = fileKey(file.repo, file.relativePath);
        const existing = persistedByFile.get(key);
        if (existing && existing.fileHash && existing.fileHash === file.contentHash) {
          reusedVectors.push(...existing.vectors);
          reusedFiles++;
          repoReusedFiles++;
          reusedChunks += existing.vectors.length;
          repoChunks += existing.vectors.length;
          totalChunks += existing.vectors.length;
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

    const filesToEmbed = changedFiles + newFiles;
    console.log(`\nReused ${reusedChunks} chunks from ${reusedFiles} unchanged files.`);
    console.log(
      `Embedding ${docsToEmbed.length} chunks from ${filesToEmbed} changed/new files (${changedFiles} changed, ${newFiles} new)...`,
    );

    const newlyEmbeddedVectors: StoredVector[] = [];
    await initializeStoredVectors(reusedVectors, config);

    if (docsToEmbed.length > 0) {
      const embeddings = createEmbeddings(config);
      const batchSize = 100;
      for (let i = 0; i < docsToEmbed.length; i += batchSize) {
        const batch = docsToEmbed.slice(i, i + batchSize);
        const texts = batch.map((d) => d.pageContent);
        const embs = await embeddings.embedDocuments(texts);
        const batchVectors: StoredVector[] = [];

        for (let j = 0; j < batch.length; j++) {
          const vector: StoredVector = {
            embedding: embs[j],
            doc: {
              pageContent: batch[j].pageContent,
              metadata: batch[j].metadata as Record<string, unknown>,
            },
          };
          batchVectors.push(vector);
          newlyEmbeddedVectors.push(vector);
        }
        await appendStoredVectors(batchVectors, config);

        if (docsToEmbed.length > batchSize) {
          process.stdout.write(
            `  Embedded ${Math.min(i + batchSize, docsToEmbed.length)}/${docsToEmbed.length}\r`,
          );
        }
      }
      if (docsToEmbed.length > batchSize) console.log();
    }

    const allDocs = [...reusedVectors, ...newlyEmbeddedVectors].map(
      (v) => new Document({ pageContent: v.doc.pageContent, metadata: v.doc.metadata }),
    );
    await persistIndex(allDocs, config);

    console.log(
      `Index complete. ${totalFiles} files scanned, ${totalChunks} chunks total (${reusedChunks} reused, ${newlyEmbeddedVectors.length} newly embedded).`,
    );
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
