// Reusable indexing pipeline.
//
// Extracted from the `index` CLI command so the MCP `oracle_reindex` verb,
// the systemd-driven background reindex, and the CLI can all share one
// code path with one set of guarantees (incremental signature reuse, orphan
// repo_meta pruning, deleted-file vector cleanup, atomic per-file upsert).

import { Document } from "@langchain/core/documents";
import { assertScanRoot, type Config } from "../config.js";
import {
  DEFAULT_MAX_FILE_SIZE_BYTES,
  discoverRepos,
  walkRepo,
  type SkippedFile,
  type WalkRepoOptions,
} from "./scanner.js";
import { mergeSkipDirs } from "./skip-dirs.js";
import { splitFile } from "./splitter.js";
import { createEmbeddings } from "../store/embeddings.js";
import { openSqliteStore, type StoredEntry } from "../store/sqlite-store.js";

export interface IndexSummary {
  reposScanned: number;
  filesScanned: number;
  filesReused: number;
  filesChanged: number;
  filesNew: number;
  filesPruned: number;
  filesSkipped: number;
  skippedFiles: Array<{
    repo: string;
    relativePath: string;
    reason: string;
    sizeBytes?: number;
    limitBytes?: number;
    message?: string;
  }>;
  chunksTotal: number;
  chunksReused: number;
  chunksEmbedded: number;
  durationMs: number;
}

export interface RunIndexOptions {
  // Per-line progress sink. Defaults to a no-op so the MCP server stays
  // quiet; the CLI passes (msg) => process.stdout.write(msg).
  logger?: (line: string) => void;
  // Per-line warning sink, same no-op-by-default pattern as `logger`. Kept
  // separate from `logger` so a caller (e.g. the CLI) can route it to
  // stderr while normal progress stays on stdout, and so zero-skip runs and
  // MCP callers stay quiet by default (nothing is ever written here unless
  // there is an actual skip to report).
  warn?: (line: string) => void;
}

function fileKey(repo: string, filePath: string): string {
  return `${repo}::${filePath}`;
}

export async function runIndex(
  config: Config,
  options: RunIndexOptions = {},
): Promise<IndexSummary> {
  assertScanRoot(config);
  const log = options.logger ?? (() => {});
  const warn = options.warn ?? (() => {});
  const startedAt = Date.now();

  log(`Scanning repos in ${config.scanRoot}...\n`);

  const repos = await discoverRepos(config.scanRoot);
  log(`Found ${repos.length} repos\n`);

  const store = openSqliteStore(config);
  try {
    store.assertCompatibleWithConfig(config);
    const orphanedMeta = store.pruneOrphanRepoMeta();
    if (orphanedMeta > 0) {
      log(`Cleared ${orphanedMeta} orphan repo_meta row(s) left over from earlier prunes.\n`);
    }
    const existingSignatures = store.fileSignatures();
    if (existingSignatures.size > 0) {
      log(
        `Loaded signatures for ${existingSignatures.size} files from ${store.dbPath} for incremental indexing\n`,
      );
    }

    const skippedFiles: IndexSummary["skippedFiles"] = [];
    const skipDirs = mergeSkipDirs(config.skipDirs);
    const walkOptions: WalkRepoOptions = {
      skipDirs,
      maxFileSizeBytes: config.maxFileSizeBytes,
      onSkip: (skip: SkippedFile) => {
        skippedFiles.push({
          repo: skip.repo,
          relativePath: skip.relativePath,
          reason: skip.reason,
          sizeBytes: skip.sizeBytes,
          limitBytes: skip.limitBytes,
          message: skip.message,
        });
      },
    };
    if (config.includeExtensions) {
      walkOptions.extensions = new Set(config.includeExtensions);
      log(
        `Using ORACLE_INCLUDE_EXTENSIONS override: ${config.includeExtensions.join(", ")}\n`,
      );
    }
    if (config.skipDirs && config.skipDirs.length > 0) {
      log(`Adding ORACLE_SKIP_DIRS to defaults: ${config.skipDirs.join(", ")}\n`);
    }
    if (config.maxFileSizeBytes !== DEFAULT_MAX_FILE_SIZE_BYTES) {
      log(
        `Using ORACLE_MAX_FILE_SIZE override: ${config.maxFileSizeBytes} bytes\n`,
      );
    }

    let filesScanned = 0;
    let filesReused = 0;
    let filesChanged = 0;
    let filesNew = 0;

    const seenKeys = new Set<string>();
    const liveRepos = new Set<string>();
    const docsToEmbed: Document[] = [];

    for (const repo of repos) {
      let repoFiles = 0;
      let repoChunks = 0;
      let repoReusedFiles = 0;
      log(`  ${repo.name}...`);

      for await (const file of walkRepo(repo.path, repo.name, config.scanRoot, walkOptions)) {
        repoFiles++;
        filesScanned++;

        const key = fileKey(file.repo, file.relativePath);
        seenKeys.add(key);
        const existing = existingSignatures.get(key);
        if (existing && existing.fileHash && existing.fileHash === file.contentHash) {
          filesReused++;
          repoReusedFiles++;
          continue;
        }

        if (existing) filesChanged++;
        else filesNew++;

        const chunks = await splitFile(file);
        docsToEmbed.push(...chunks);
        repoChunks += chunks.length;
      }

      if (repoFiles > 0) liveRepos.add(repo.name);

      log(` ${repoFiles} files, ${repoChunks} chunks (${repoReusedFiles} files reused)\n`);
    }

    // Loud, opt-in reporting of per-file skips (too-large / read-error).
    // Both reasons mean a file that would otherwise have entered the index
    // did not — the exact silent-drop failure mode this feature closes. Kept
    // on a separate `warn` sink (defaults to a no-op) so the MCP path and
    // zero-skip CLI runs stay exactly as quiet as before.
    if (skippedFiles.length > 0) {
      for (const skip of skippedFiles) {
        if (skip.reason === "too-large") {
          warn(
            `WARNING: skipped ${skip.relativePath} — ${skip.sizeBytes} bytes > ORACLE_MAX_FILE_SIZE=${skip.limitBytes}\n`,
          );
        } else {
          warn(`WARNING: skipped ${skip.relativePath} — read error: ${skip.message}\n`);
        }
      }
      warn(
        `WARNING: ${skippedFiles.length} file(s) skipped during scan; raise ORACLE_MAX_FILE_SIZE to index larger files.\n`,
      );
    }

    // Files that existed in the store but were not seen this scan → deleted
    // on disk. Drop their vectors so stale chunks don't linger.
    let filesPruned = 0;
    for (const [key, sig] of existingSignatures) {
      if (seenKeys.has(key)) continue;
      const removed = store.deleteByFile(sig.repo, sig.filePath);
      if (removed > 0) filesPruned++;
    }
    if (filesPruned > 0) {
      log(`Pruned ${filesPruned} files that vanished from disk.\n`);
    }

    // Stamp every repo that still has at least one file on disk so reused-
    // only repos still advance last_indexed_at. upsertFile/insertBatch
    // later in this run touch the same row again with a slightly newer
    // timestamp for repos that did pick up changes — last write wins,
    // which is what we want.
    //
    // Deliberately excluding repos that were discovered but yielded zero
    // files this scan (e.g. an entire repo was deleted between runs).
    const scannedAt = new Date().toISOString();
    for (const repoName of liveRepos) {
      store.touchRepo(repoName, scannedAt);
    }

    const filesToEmbed = filesChanged + filesNew;
    const countBeforeEmbed = store.count();

    log(
      `\nEmbedding ${docsToEmbed.length} chunks from ${filesToEmbed} changed/new files (${filesChanged} changed, ${filesNew} new). ${filesReused} files reused.\n`,
    );

    if (docsToEmbed.length === 0) {
      const chunksTotal = countBeforeEmbed;
      log(
        `Index complete. ${filesScanned} files scanned, ${chunksTotal} chunks total (${chunksTotal} reused, 0 newly embedded).\n`,
      );
      return {
        reposScanned: repos.length,
        filesScanned,
        filesReused,
        filesChanged,
        filesNew,
        filesPruned,
        filesSkipped: skippedFiles.length,
        skippedFiles,
        chunksTotal,
        chunksReused: chunksTotal,
        chunksEmbedded: 0,
        durationMs: Date.now() - startedAt,
      };
    }

    // Initialise schema now that we know the embedding dimension (run the
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
        docsByFile.set(key, { repo: metadata.repo, filePath: metadata.filePath, docs: [doc] });
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
        log(`  Embedded ${Math.min(i + batchSize, rest.length) + 1}/${docsToEmbed.length}\r`);
      }
    }
    if (rest.length > batchSize) log("\n");

    // Transactionally upsert each file. upsertFile removes stale chunks for
    // that (repo, filePath) first, so changed files swap cleanly.
    for (const [key, entries] of embeddedByKey) {
      const group = docsByFile.get(key)!;
      const contentHash =
        (group.docs[0]?.metadata as { fileHash?: string })?.fileHash ?? null;
      store.upsertFile(group.repo, group.filePath, contentHash, entries);
    }

    const finalTotal = store.count();
    const chunksReused = finalTotal - docsToEmbed.length;
    log(
      `Index complete. ${filesScanned} files scanned, ${finalTotal} chunks total (${chunksReused} reused, ${docsToEmbed.length} newly embedded).\n`,
    );
    return {
      reposScanned: repos.length,
      filesScanned,
      filesReused,
      filesChanged,
      filesNew,
      filesPruned,
      filesSkipped: skippedFiles.length,
      skippedFiles,
      chunksTotal: finalTotal,
      chunksReused,
      chunksEmbedded: docsToEmbed.length,
      durationMs: Date.now() - startedAt,
    };
  } finally {
    store.close();
  }
}

export function formatIndexSummary(summary: IndexSummary): string {
  const seconds = (summary.durationMs / 1000).toFixed(1);
  return [
    `Reindex complete in ${seconds}s.`,
    `Repos: ${summary.reposScanned}, files: ${summary.filesScanned} scanned (`
      + `${summary.filesReused} reused, ${summary.filesChanged} changed, `
      + `${summary.filesNew} new, ${summary.filesPruned} pruned, `
      + `${summary.filesSkipped} skipped).`,
    `Chunks: ${summary.chunksTotal} total (`
      + `${summary.chunksReused} reused, ${summary.chunksEmbedded} embedded).`,
  ].join("\n");
}
