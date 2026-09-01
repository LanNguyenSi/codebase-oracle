import { afterEach, describe, it, expect } from "vitest";
import { mkdtemp, rm, writeFile, mkdir, chmod } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import Database from "better-sqlite3";
import * as sqliteVec from "sqlite-vec";

// Spawns the real `tsx src/index.ts index --path <tmp>` command with the
// stub embedding provider so we exercise the full orchestration:
// discoverRepos → walkRepo → prune-by-file → liveRepos gating → touchRepo.
// The store-API regression test in tests/unit/sqlite-store.test.ts replays
// the sequence by hand; this one drives the actual binary so a refactor
// that, say, flips liveRepos back to scannedRepos would be caught here.

const repoRoot = fileURLToPath(new URL("../..", import.meta.url));
const indexEntry = join(repoRoot, "src", "index.ts");

const tmpDirs: string[] = [];
async function makeTmpDir(): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), "oracle-cli-"));
  tmpDirs.push(dir);
  return dir;
}

afterEach(async () => {
  while (tmpDirs.length > 0) {
    const dir = tmpDirs.pop();
    if (dir) await rm(dir, { recursive: true, force: true });
  }
});

async function makeRepo(scanRoot: string, name: string, files: Record<string, string>): Promise<void> {
  const repoDir = join(scanRoot, name);
  await mkdir(join(repoDir, ".git"), { recursive: true });
  for (const [rel, content] of Object.entries(files)) {
    const abs = join(repoDir, rel);
    await mkdir(join(abs, ".."), { recursive: true });
    await writeFile(abs, content, "utf8");
  }
}

interface MetaRow {
  repo: string;
  last_indexed_at: string | null;
}

interface SkipMetaRow {
  repo: string;
  size_count: number;
  error_count: number;
  examples: string;
}

function readStore(dataDir: string): {
  repos: Array<{ repo: string; chunks: number }>;
  meta: MetaRow[];
  skipMeta: SkipMetaRow[];
  filesByRepo: Map<string, string[]>;
} {
  const db = new Database(join(dataDir, "store.db"), { readonly: true });
  sqliteVec.load(db);
  const repos = db
    .prepare("SELECT repo, COUNT(*) AS chunks FROM docs GROUP BY repo ORDER BY repo")
    .all() as Array<{ repo: string; chunks: number }>;
  const meta = db
    .prepare("SELECT repo, last_indexed_at FROM repo_meta ORDER BY repo")
    .all() as MetaRow[];
  const skipMeta = db
    .prepare("SELECT repo, size_count, error_count, examples FROM repo_skip_meta ORDER BY repo")
    .all() as SkipMetaRow[];
  const fileRows = db
    .prepare("SELECT DISTINCT repo, file_path FROM docs ORDER BY repo, file_path")
    .all() as Array<{ repo: string; file_path: string }>;
  const filesByRepo = new Map<string, string[]>();
  for (const row of fileRows) {
    const list = filesByRepo.get(row.repo) ?? [];
    list.push(row.file_path);
    filesByRepo.set(row.repo, list);
  }
  db.close();
  return { repos, meta, skipMeta, filesByRepo };
}

function runListRepos(
  dataDir: string,
): { stdout: string; stderr: string; status: number | null } {
  const result = spawnSync("npx", ["tsx", indexEntry, "list-repos"], {
    encoding: "utf8",
    cwd: repoRoot,
    env: {
      ...process.env,
      ORACLE_DATA_DIR: dataDir,
      ORACLE_EMBEDDING_PROVIDER: "stub",
      ORACLE_EMBEDDING_MODEL: "stub",
    },
  });
  return { stdout: result.stdout ?? "", stderr: result.stderr ?? "", status: result.status };
}

function runIndex(
  scanRoot: string,
  dataDir: string,
  extraEnv: Record<string, string> = {},
): { stdout: string; stderr: string; status: number | null } {
  const result = spawnSync("npx", ["tsx", indexEntry, "index", "--path", scanRoot], {
    encoding: "utf8",
    cwd: repoRoot,
    env: {
      ...process.env,
      ORACLE_DATA_DIR: dataDir,
      ORACLE_EMBEDDING_PROVIDER: "stub",
      ORACLE_EMBEDDING_MODEL: "stub",
      ORACLE_SCAN_ROOT: scanRoot,
      ...extraEnv,
    },
  });
  return { stdout: result.stdout ?? "", stderr: result.stderr ?? "", status: result.status };
}

describe("oracle index CLI integration", () => {
  it(
    "drops vanished repos from repo_meta and advances last_indexed_at on reused-only repos",
    { timeout: 30_000 },
    async () => {
      const tmp = await makeTmpDir();
      const scanRoot = join(tmp, "repos");
      const dataDir = join(tmp, "data");
      await mkdir(scanRoot, { recursive: true });

      await makeRepo(scanRoot, "kept", {
        "a.ts": "export const kept_a = 1;\n",
        "b.ts": "export const kept_b = 2;\n",
      });
      await makeRepo(scanRoot, "vanished", {
        "x.ts": "export const vanished_x = 1;\n",
      });
      await makeRepo(scanRoot, "partial", {
        "stays.ts": "export const partial_stays = 1;\n",
        "goes.ts": "export const partial_goes = 1;\n",
      });

      const first = runIndex(scanRoot, dataDir);
      expect(first.status, `first index failed: ${first.stderr}`).toBe(0);
      expect(first.stdout).toMatch(/Found 3 repos/);

      const after1 = readStore(dataDir);
      expect(after1.repos.map((r) => r.repo)).toEqual(["kept", "partial", "vanished"]);
      expect(after1.meta.map((m) => m.repo)).toEqual(["kept", "partial", "vanished"]);
      const stamp1 = after1.meta.find((m) => m.repo === "kept")!.last_indexed_at!;
      expect(stamp1).toMatch(/^\d{4}-\d{2}-\d{2}T/);

      // Mutate disk: drop one full repo and one file from another.
      await rm(join(scanRoot, "vanished"), { recursive: true, force: true });
      await rm(join(scanRoot, "partial", "goes.ts"));
      // Sleep so the new `scannedAt` ISO string is strictly newer than stamp1.
      await new Promise((r) => setTimeout(r, 1100));

      const second = runIndex(scanRoot, dataDir);
      expect(second.status, `second index failed: ${second.stderr}`).toBe(0);

      const after2 = readStore(dataDir);
      // vanished gone from BOTH docs and repo_meta — no orphans.
      expect(after2.repos.map((r) => r.repo)).toEqual(["kept", "partial"]);
      expect(after2.meta.map((m) => m.repo)).toEqual(["kept", "partial"]);
      // kept had zero changes but its freshness still advanced (this is the
      // bug-2 regression: reused-only repos must still be touched).
      const stamp2 = after2.meta.find((m) => m.repo === "kept")!.last_indexed_at!;
      expect(new Date(stamp2).getTime()).toBeGreaterThan(new Date(stamp1).getTime());
      // partial still has its surviving file; goes.ts is gone.
      expect(after2.filesByRepo.get("partial")).toEqual(["partial/stays.ts"]);
      expect(second.stdout).toMatch(/Pruned \d+ files? that vanished from disk\./);
    },
  );

  it("startup sweep clears orphan repo_meta rows from a legacy store", { timeout: 30_000 }, async () => {
    const tmp = await makeTmpDir();
    const scanRoot = join(tmp, "repos");
    const dataDir = join(tmp, "data");
    await mkdir(dataDir, { recursive: true });
    await mkdir(scanRoot, { recursive: true });
    await makeRepo(scanRoot, "kept", { "a.ts": "export const kept = 1;\n" });

    // Seed a populated store, then mutate it into the pre-fix state: docs
    // for `legacy` are gone but its repo_meta row survives. This is exactly
    // the situation real stores carried before the deleteByFile fix.
    const first = runIndex(scanRoot, dataDir);
    expect(first.status, first.stderr).toBe(0);

    {
      const db = new Database(join(dataDir, "store.db"));
      sqliteVec.load(db);
      db.prepare("INSERT INTO repo_meta(repo, last_indexed_at) VALUES('legacy-orphan', ?)")
        .run(new Date().toISOString());
      db.close();
    }

    const beforeSweep = readStore(dataDir);
    expect(beforeSweep.meta.map((m) => m.repo)).toContain("legacy-orphan");

    const second = runIndex(scanRoot, dataDir);
    expect(second.status, second.stderr).toBe(0);
    expect(second.stdout).toMatch(/Cleared 1 orphan repo_meta row/);

    const afterSweep = readStore(dataDir);
    expect(afterSweep.meta.map((m) => m.repo)).not.toContain("legacy-orphan");
  });

  it(
    "reports an oversized file loudly on stderr, keeps it out of the store, and still indexes its sibling",
    { timeout: 30_000 },
    async () => {
      const tmp = await makeTmpDir();
      const scanRoot = join(tmp, "repos");
      const dataDir = join(tmp, "data");
      await mkdir(scanRoot, { recursive: true });

      // 2000 bytes comfortably clears the ORACLE_MAX_FILE_SIZE=500 ceiling
      // set for this run below (real repro was tasks.ts at 207,716 bytes
      // vs. the old hardcoded 200_000; a smaller number keeps the fixture
      // cheap while exercising the exact same code path).
      const oversized = "x".repeat(2000) + "\n";
      await makeRepo(scanRoot, "oversized", {
        "big.ts": oversized,
        "small.ts": "export const small = 1;\n",
      });

      const result = runIndex(scanRoot, dataDir, { ORACLE_MAX_FILE_SIZE: "500" });
      expect(result.status, `index failed: ${result.stderr}`).toBe(0);

      // Loud per-file skip line naming the path, size, and configured limit.
      expect(result.stderr).toMatch(
        /WARNING: skipped oversized\/big\.ts — \d+ bytes > ORACLE_MAX_FILE_SIZE=500/,
      );
      // Run summary line (the acceptance criterion: never a silent drop).
      expect(result.stderr).toMatch(/WARNING: 1 file\(s\) skipped during scan/);

      const store = readStore(dataDir);
      // The oversized file never entered the store; its sibling did.
      expect(store.filesByRepo.get("oversized")).toEqual(["oversized/small.ts"]);
    },
  );

  it(
    "prunes a previously-indexed file that a lowered ORACLE_MAX_FILE_SIZE now skips",
    { timeout: 30_000 },
    async () => {
      const tmp = await makeTmpDir();
      const scanRoot = join(tmp, "repos");
      const dataDir = join(tmp, "data");
      await mkdir(scanRoot, { recursive: true });

      await makeRepo(scanRoot, "shrinking", {
        "big.ts": "x".repeat(2000) + "\n",
        "small.ts": "export const small = 1;\n",
      });

      // First run under the default limit: both files enter the store.
      const first = runIndex(scanRoot, dataDir);
      expect(first.status, `first index failed: ${first.stderr}`).toBe(0);
      expect(readStore(dataDir).filesByRepo.get("shrinking")).toEqual([
        "shrinking/big.ts",
        "shrinking/small.ts",
      ]);

      // Second run with a lowered limit: big.ts is now over the ceiling. It
      // must be reported AND its stale chunks pruned by the deleted-file
      // sweep — a skipped file is not in seenKeys, so lowering the limit
      // must not leave its old vectors lingering in the store.
      const second = runIndex(scanRoot, dataDir, { ORACLE_MAX_FILE_SIZE: "500" });
      expect(second.status, `second index failed: ${second.stderr}`).toBe(0);
      expect(second.stderr).toMatch(
        /WARNING: skipped shrinking\/big\.ts — \d+ bytes > ORACLE_MAX_FILE_SIZE=500/,
      );
      expect(second.stdout).toMatch(/Pruned 1 files? that vanished from disk\./);
      expect(readStore(dataDir).filesByRepo.get("shrinking")).toEqual([
        "shrinking/small.ts",
      ]);
    },
  );

  it(
    "indexes a 600 KB markdown file under the default config, and list-repos shows no skipped files",
    { timeout: 30_000 },
    async () => {
      // 600 KB clears the general ORACLE_MAX_FILE_SIZE=500_000 default that
      // this exact repro was reported against (a repo's CHANGELOG.md), but
      // must stay under maxTextFileSizeBytes' default (2_000_000) so the
      // per-type ceiling — not just a raised general default — is what's
      // under test.
      const tmp = await makeTmpDir();
      const scanRoot = join(tmp, "repos");
      const dataDir = join(tmp, "data");
      await mkdir(scanRoot, { recursive: true });

      const bigMarkdown = "# Changelog\n\n" + "x".repeat(600_000) + "\n";
      await makeRepo(scanRoot, "docs-heavy", {
        "CHANGELOG.md": bigMarkdown,
        "small.ts": "export const small = 1;\n",
      });

      const result = runIndex(scanRoot, dataDir);
      expect(result.status, `index failed: ${result.stderr}`).toBe(0);
      expect(result.stderr).not.toMatch(/WARNING: skipped/);

      const store = readStore(dataDir);
      expect(store.filesByRepo.get("docs-heavy")).toEqual([
        "docs-heavy/CHANGELOG.md",
        "docs-heavy/small.ts",
      ]);

      const listed = runListRepos(dataDir);
      expect(listed.status, `list-repos failed: ${listed.stderr}`).toBe(0);
      expect(listed.stdout).not.toContain("skipped");
    },
  );

  it(
    "list-repos (CLI) and repo_skip_meta report a per-repo skipped-file count that reflects only the LAST run",
    { timeout: 30_000 },
    async () => {
      const tmp = await makeTmpDir();
      const scanRoot = join(tmp, "repos");
      const dataDir = join(tmp, "data");
      await mkdir(scanRoot, { recursive: true });

      await makeRepo(scanRoot, "flaky", {
        "big.ts": "x".repeat(2000) + "\n",
        "small.ts": "export const small = 1;\n",
      });

      // First run: big.ts is skipped under a lowered ceiling.
      const first = runIndex(scanRoot, dataDir, { ORACLE_MAX_FILE_SIZE: "500" });
      expect(first.status, `first index failed: ${first.stderr}`).toBe(0);

      const afterFirst = readStore(dataDir);
      expect(afterFirst.skipMeta).toEqual([
        { repo: "flaky", size_count: 1, error_count: 0, examples: JSON.stringify(["flaky/big.ts"]) },
      ]);

      const listedFirst = runListRepos(dataDir);
      expect(listedFirst.status, listedFirst.stderr).toBe(0);
      expect(listedFirst.stdout).toMatch(/flaky.*1 file\(s\) skipped in the last index run/);

      // Second run under the default (unrestricted) ceiling: big.ts now
      // fits, nothing is skipped. The stored count must go back to 0, not
      // stay stuck at 1 from the first run.
      const second = runIndex(scanRoot, dataDir);
      expect(second.status, `second index failed: ${second.stderr}`).toBe(0);

      const afterSecond = readStore(dataDir);
      expect(afterSecond.skipMeta).toEqual([
        { repo: "flaky", size_count: 0, error_count: 0, examples: "[]" },
      ]);

      const listedSecond = runListRepos(dataDir);
      expect(listedSecond.status, listedSecond.stderr).toBe(0);
      expect(listedSecond.stdout).not.toContain("skipped");
    },
  );

  it(
    "a repo whose only file blows the size ceiling has zero docs but still appears in list-repos with its skip count",
    { timeout: 30_000 },
    async () => {
      // Reproduces the reviewer's finding verbatim: a scan root containing a
      // repo with a single oversized file persists a repo_skip_meta row
      // (tally {size_count: 1}) but has NO docs row at all, since nothing of
      // it ever entered the store. Before the listRepos widening this repo
      // was entirely absent from list-repos output.
      const tmp = await makeTmpDir();
      const scanRoot = join(tmp, "repos");
      const dataDir = join(tmp, "data");
      await mkdir(scanRoot, { recursive: true });

      await makeRepo(scanRoot, "allbig", {
        "big.ts": "x".repeat(2000) + "\n",
      });
      await makeRepo(scanRoot, "normal", {
        "a.ts": "export const a = 1;\n",
      });

      const result = runIndex(scanRoot, dataDir, { ORACLE_MAX_FILE_SIZE: "500" });
      expect(result.status, `index failed: ${result.stderr}`).toBe(0);

      const store = readStore(dataDir);
      // allbig has no docs at all, the exact case the widened query covers.
      expect(store.filesByRepo.get("allbig")).toBeUndefined();
      expect(store.repos.map((r) => r.repo)).toEqual(["normal"]);
      expect(store.skipMeta).toContainEqual({
        repo: "allbig",
        size_count: 1,
        error_count: 0,
        examples: JSON.stringify(["allbig/big.ts"]),
      });

      const listed = runListRepos(dataDir);
      expect(listed.status, `list-repos failed: ${listed.stderr}`).toBe(0);
      expect(listed.stdout).toMatch(
        /allbig — 0 chunks across 0 files; 1 file\(s\) skipped in the last index run \(1 too large; e\.g\. allbig\/big\.ts\)/,
      );
    },
  );

  it(
    "a repo deleted from the scan root loses its repo_skip_meta row on the next run",
    { timeout: 30_000 },
    async () => {
      const tmp = await makeTmpDir();
      const scanRoot = join(tmp, "repos");
      const dataDir = join(tmp, "data");
      await mkdir(scanRoot, { recursive: true });

      await makeRepo(scanRoot, "kept", { "a.ts": "export const a = 1;\n" });
      await makeRepo(scanRoot, "goneSoon", { "big.ts": "x".repeat(2000) + "\n" });

      const first = runIndex(scanRoot, dataDir, { ORACLE_MAX_FILE_SIZE: "500" });
      expect(first.status, `first index failed: ${first.stderr}`).toBe(0);
      expect(readStore(dataDir).skipMeta.map((s) => s.repo)).toEqual(["goneSoon", "kept"]);

      // Delete the repo that skipped a file entirely: its repo_skip_meta
      // row must not survive forever like the pre-fix repo_meta orphans did.
      await rm(join(scanRoot, "goneSoon"), { recursive: true, force: true });

      const second = runIndex(scanRoot, dataDir, { ORACLE_MAX_FILE_SIZE: "500" });
      expect(second.status, `second index failed: ${second.stderr}`).toBe(0);
      expect(second.stdout).toMatch(/Cleared 1 orphan repo_skip_meta row/);

      const afterSecond = readStore(dataDir);
      expect(afterSecond.skipMeta.map((s) => s.repo)).toEqual(["kept"]);
    },
  );

  it(
    "caps persisted skip examples at SKIP_EXAMPLES_LIMIT (5) even when more files were skipped",
    { timeout: 30_000 },
    async () => {
      const tmp = await makeTmpDir();
      const scanRoot = join(tmp, "repos");
      const dataDir = join(tmp, "data");
      await mkdir(scanRoot, { recursive: true });

      const files: Record<string, string> = {};
      for (let i = 0; i < 7; i++) {
        files[`big${i}.ts`] = "x".repeat(2000) + "\n";
      }
      await makeRepo(scanRoot, "manybig", files);

      const result = runIndex(scanRoot, dataDir, { ORACLE_MAX_FILE_SIZE: "500" });
      expect(result.status, `index failed: ${result.stderr}`).toBe(0);

      const skipMeta = readStore(dataDir).skipMeta.find((s) => s.repo === "manybig")!;
      expect(skipMeta.size_count).toBe(7);
      const examples = JSON.parse(skipMeta.examples) as string[];
      expect(examples).toHaveLength(5);

      const listed = runListRepos(dataDir);
      expect(listed.status, listed.stderr).toBe(0);
      expect(listed.stdout).toMatch(/7 file\(s\) skipped in the last index run \(7 too large;/);
    },
  );

  it(
    "a real unreadable file (chmod 000) is skipped as a read-error and counted end to end",
    { timeout: 30_000 },
    async () => {
      // Root (and some CI containers) bypass file permission bits entirely,
      // which would make this test silently pass for the wrong reason;
      // skip it rather than assert a false positive.
      if (process.getuid && process.getuid() === 0) {
        return;
      }
      const tmp = await makeTmpDir();
      const scanRoot = join(tmp, "repos");
      const dataDir = join(tmp, "data");
      await mkdir(scanRoot, { recursive: true });

      await makeRepo(scanRoot, "locked", {
        "readable.ts": "export const readable = 1;\n",
        "secret.ts": "export const secret = 1;\n",
      });
      await chmod(join(scanRoot, "locked", "secret.ts"), 0o000);

      try {
        const result = runIndex(scanRoot, dataDir);
        expect(result.status, `index failed: ${result.stderr}`).toBe(0);
        expect(result.stderr).toMatch(/WARNING: skipped locked\/secret\.ts — read error:/);

        const store = readStore(dataDir);
        expect(store.filesByRepo.get("locked")).toEqual(["locked/readable.ts"]);
        expect(store.skipMeta).toContainEqual({
          repo: "locked",
          size_count: 0,
          error_count: 1,
          examples: JSON.stringify(["locked/secret.ts"]),
        });
      } finally {
        // Restore permissions so the afterEach rm() can clean up the tree.
        await chmod(join(scanRoot, "locked", "secret.ts"), 0o644);
      }
    },
  );
});
