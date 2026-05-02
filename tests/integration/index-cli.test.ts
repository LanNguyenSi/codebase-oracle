import { afterEach, describe, it, expect } from "vitest";
import { mkdtemp, rm, writeFile, mkdir } from "node:fs/promises";
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

function readStore(dataDir: string): {
  repos: Array<{ repo: string; chunks: number }>;
  meta: MetaRow[];
} {
  const db = new Database(join(dataDir, "store.db"), { readonly: true });
  sqliteVec.load(db);
  const repos = db
    .prepare("SELECT repo, COUNT(*) AS chunks FROM docs GROUP BY repo ORDER BY repo")
    .all() as Array<{ repo: string; chunks: number }>;
  const meta = db
    .prepare("SELECT repo, last_indexed_at FROM repo_meta ORDER BY repo")
    .all() as MetaRow[];
  db.close();
  return { repos, meta };
}

function runIndex(scanRoot: string, dataDir: string): { stdout: string; stderr: string; status: number | null } {
  const result = spawnSync("npx", ["tsx", indexEntry, "index", "--path", scanRoot], {
    encoding: "utf8",
    cwd: repoRoot,
    env: {
      ...process.env,
      ORACLE_DATA_DIR: dataDir,
      ORACLE_EMBEDDING_PROVIDER: "stub",
      ORACLE_EMBEDDING_MODEL: "stub",
      ORACLE_SCAN_ROOT: scanRoot,
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
      expect(after2.repos.find((r) => r.repo === "partial")!.chunks).toBeGreaterThan(0);
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
});
