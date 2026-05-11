// Retrieval-quality eval runner for codebase-oracle.
//
// Indexes the vendored fixture corpus under tests/eval/corpus/, runs
// every labelled question in tests/eval/questions.json through
// oracle_search, asserts each expected file path appears within its
// rank bucket, and compares the pass set against tests/eval/baseline.json.
// A regression (a question that was passing in the baseline but fails
// now) exits non-zero so a CI gate can fail loudly.
//
// Usage:
//   npm run eval                    -- run + compare against baseline
//   npm run eval -- --update        -- overwrite baseline with the
//                                      current results (use after an
//                                      intentional improvement)
//
// Requires real embeddings (OPENAI_API_KEY by default, or Ollama via
// ORACLE_EMBEDDING_PROVIDER=ollama). The eval uses a private
// ORACLE_DATA_DIR under tests/eval/.cache so it never touches the
// user's main index.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadEnvFromFile } from "../../src/env.js";
import { loadConfig } from "../../src/config.js";
import { runIndex } from "../../src/ingest/runner.js";
import { createEmbeddings } from "../../src/store/embeddings.js";
import { createVectorStore } from "../../src/store/vector-store.js";
import { searchCodebase } from "../../src/retrieval/chain.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const CORPUS_ROOT = path.join(__dirname, "corpus");
const DATA_DIR = path.join(__dirname, ".cache");
const QUESTIONS_PATH = path.join(__dirname, "questions.json");
const BASELINE_PATH = path.join(__dirname, "baseline.json");

type RankBucket = `top-${number}`;

interface ExpectedFilePath {
  path: string;
  atRank: RankBucket;
}

interface Question {
  id: string;
  question: string;
  expectedFilePaths: ExpectedFilePath[];
}

interface CheckResult {
  path: string;
  atRank: RankBucket;
  pass: boolean;
  actualRank: number | null;
}

interface QuestionResult {
  id: string;
  pass: boolean;
  checks: CheckResult[];
}

interface Baseline {
  generatedAt: string;
  results: Array<{ id: string; pass: boolean }>;
}

function rankLimit(bucket: RankBucket): number {
  const n = parseInt(bucket.replace("top-", ""), 10);
  if (!Number.isFinite(n) || n < 1) {
    throw new Error(`Invalid rank bucket: ${bucket}`);
  }
  return n;
}

async function runEval(): Promise<{ results: QuestionResult[]; regressions: string[] }> {
  loadEnvFromFile();

  // Fresh state every run: nuke the eval cache so corpus + embedding
  // model changes always rebuild deterministically.
  if (fs.existsSync(DATA_DIR)) {
    fs.rmSync(DATA_DIR, { recursive: true, force: true });
  }
  fs.mkdirSync(DATA_DIR, { recursive: true });

  const config = loadConfig({ scanRoot: CORPUS_ROOT, dataDir: DATA_DIR });

  process.stderr.write(`[eval] indexing ${CORPUS_ROOT}\n`);
  await runIndex(config, {
    logger: (line) => process.stderr.write(`[index] ${line}`),
  });

  const embeddings = createEmbeddings(config);
  const store = await createVectorStore(embeddings, config);
  try {
    const questions: Question[] = JSON.parse(fs.readFileSync(QUESTIONS_PATH, "utf-8"));
    const results: QuestionResult[] = [];

    for (const q of questions) {
      // Pull a generous slice so a top-10 bucket can still resolve when
      // the embedder happens to surface unrelated chunks first.
      const docs = await searchCodebase(q.question, store, { limit: 15 });
      const ranks = new Map<string, number>();
      docs.forEach((doc, i) => {
        const filePath = (doc.metadata as { filePath?: string }).filePath ?? "";
        if (!ranks.has(filePath)) ranks.set(filePath, i + 1);
      });

      const checks: CheckResult[] = q.expectedFilePaths.map((expected) => {
        const actualRank = ranks.get(expected.path) ?? null;
        const limit = rankLimit(expected.atRank);
        const pass = actualRank !== null && actualRank <= limit;
        return { path: expected.path, atRank: expected.atRank, pass, actualRank };
      });

      results.push({
        id: q.id,
        pass: checks.every((c) => c.pass),
        checks,
      });
    }

    // Compare against baseline.
    const regressions: string[] = [];
    if (fs.existsSync(BASELINE_PATH)) {
      const baseline: Baseline = JSON.parse(fs.readFileSync(BASELINE_PATH, "utf-8"));
      for (const r of results) {
        const previous = baseline.results.find((b) => b.id === r.id);
        if (previous?.pass && !r.pass) {
          regressions.push(r.id);
        }
      }
    }

    return { results, regressions };
  } finally {
    store.close();
  }
}

function renderReport(results: QuestionResult[], regressions: string[]): void {
  for (const r of results) {
    const summary = r.pass ? "PASS" : "FAIL";
    process.stdout.write(`  [${summary}] ${r.id}\n`);
    for (const c of r.checks) {
      const actual = c.actualRank === null ? "missing" : `rank ${c.actualRank}`;
      const verdict = c.pass ? "ok" : "fail";
      process.stdout.write(
        `        ${verdict}: ${c.path} expected ${c.atRank}, got ${actual}\n`,
      );
    }
  }

  const passed = results.filter((r) => r.pass).length;
  const total = results.length;
  const pct = total === 0 ? 0 : Math.round((passed / total) * 100);
  process.stdout.write(
    `\neval: ${passed}/${total} questions met expectations (${pct}%); ` +
      `regressions vs. baseline: ${regressions.length}\n`,
  );
  if (regressions.length > 0) {
    process.stdout.write(`regressed: ${regressions.join(", ")}\n`);
  }
}

function writeBaseline(results: QuestionResult[]): void {
  const baseline: Baseline = {
    generatedAt: new Date().toISOString(),
    results: results.map((r) => ({ id: r.id, pass: r.pass })),
  };
  fs.writeFileSync(BASELINE_PATH, JSON.stringify(baseline, null, 2) + "\n", "utf-8");
  process.stdout.write(`baseline updated: ${BASELINE_PATH}\n`);
}

async function main(): Promise<void> {
  const updateBaseline = process.argv.includes("--update");

  const { results, regressions } = await runEval();
  renderReport(results, regressions);

  if (updateBaseline) {
    writeBaseline(results);
    process.exit(0);
  }

  if (regressions.length > 0) {
    process.exit(1);
  }
}

main().catch((err) => {
  process.stderr.write(`eval runner failed: ${err instanceof Error ? err.message : String(err)}\n`);
  if (err instanceof Error && err.stack) {
    process.stderr.write(err.stack + "\n");
  }
  process.exit(2);
});
