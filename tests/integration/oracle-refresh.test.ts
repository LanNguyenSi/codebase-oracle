import { afterEach, describe, it, expect } from "vitest";
import { existsSync } from "node:fs";
import { mkdtemp, rm, writeFile, mkdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { execFile, execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";

// Drives the real scripts/oracle-refresh.sh over throwaway local git repos:
// a bare "origin", two clones (A stays clean and behind, B goes dirty), and
// a plain non-repo directory that must be ignored. Confirms the pull/skip
// classification, the summary line, and that the (stubbed) index command
// always runs afterward.

const repoRoot = fileURLToPath(new URL("../..", import.meta.url));
const scriptPath = join(repoRoot, "scripts", "oracle-refresh.sh");

const tmpDirs: string[] = [];
async function makeTmpDir(prefix: string): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), prefix));
  tmpDirs.push(dir);
  return dir;
}

afterEach(async () => {
  while (tmpDirs.length > 0) {
    const dir = tmpDirs.pop();
    if (dir) await rm(dir, { recursive: true, force: true });
  }
});

function git(dir: string, args: string[]): string {
  return execFileSync("git", args, { cwd: dir, encoding: "utf8" }).trim();
}

function configureIdentity(dir: string): void {
  git(dir, ["config", "user.email", "oracle-refresh-test@example.invalid"]);
  git(dir, ["config", "user.name", "Oracle Refresh Test"]);
}

interface Scenario {
  scanRoot: string;
  dataDir: string;
  originGit: string;
  aDir: string;
  bDir: string;
  markerPath: string;
}

async function setupScenario(): Promise<Scenario> {
  const workDir = await makeTmpDir("oracle-refresh-work-");
  const scanRoot = await makeTmpDir("oracle-refresh-scan-");
  const dataDir = await makeTmpDir("oracle-refresh-data-");

  const originGit = join(workDir, "origin.git");
  execFileSync("git", ["init", "--quiet", "--bare", originGit]);
  // Pin the bare repo's default branch so clones don't depend on the
  // machine's global init.defaultBranch setting.
  execFileSync("git", ["symbolic-ref", "HEAD", "refs/heads/master"], { cwd: originGit });

  const seedDir = join(workDir, "seed");
  execFileSync("git", ["clone", "--quiet", originGit, seedDir]);
  configureIdentity(seedDir);
  await writeFile(join(seedDir, "file1.txt"), "one\n", "utf8");
  git(seedDir, ["add", "file1.txt"]);
  git(seedDir, ["commit", "-q", "-m", "commit 1"]);
  git(seedDir, ["push", "-q", "-u", "origin", "master"]);

  const aDir = join(scanRoot, "A");
  execFileSync("git", ["clone", "--quiet", originGit, aDir]);
  configureIdentity(aDir);

  const bDir = join(scanRoot, "B");
  execFileSync("git", ["clone", "--quiet", originGit, bDir]);
  configureIdentity(bDir);

  // Advance origin beyond both clones' current HEAD.
  await writeFile(join(seedDir, "file2.txt"), "two\n", "utf8");
  git(seedDir, ["add", "file2.txt"]);
  git(seedDir, ["commit", "-q", "-m", "commit 2"]);
  git(seedDir, ["push", "-q", "origin", "master"]);

  // B is dirty via an untracked file. Untracked (not modified-tracked) so a
  // pull would not itself conflict; the script must still skip it purely on
  // the "checkout is dirty" policy.
  await writeFile(join(bDir, "scratch.txt"), "scratch\n", "utf8");

  // A plain directory with no .git must be ignored entirely.
  const notARepo = join(scanRoot, "not-a-repo");
  await mkdir(notARepo, { recursive: true });
  await writeFile(join(notARepo, "readme.txt"), "not a repo\n", "utf8");

  const markerPath = join(workDir, "indexed");

  return { scanRoot, dataDir, originGit, aDir, bDir, markerPath };
}

function runScript(
  scenario: Scenario,
  extraEnv: Record<string, string> = {},
): Promise<{ stdout: string; stderr: string; code: number }> {
  return new Promise((resolvePromise) => {
    execFile(
      "bash",
      [scriptPath],
      {
        cwd: repoRoot,
        env: {
          ...process.env,
          ORACLE_SCAN_ROOT: scenario.scanRoot,
          ORACLE_DATA_DIR: scenario.dataDir,
          ORACLE_REFRESH_INDEX_CMD: `touch "${scenario.markerPath}"`,
          ...extraEnv,
        },
      },
      (error, stdout, stderr) => {
        const code = error
          ? (typeof (error as NodeJS.ErrnoException & { code?: unknown }).code === "number"
            ? (error as unknown as { code: number }).code
            : 1)
          : 0;
        resolvePromise({ stdout, stderr, code });
      },
    );
  });
}

describe("scripts/oracle-refresh.sh", () => {
  it(
    "pulls a clean tracked repo, skips a dirty one, ignores a non-repo dir, and runs the index command",
    { timeout: 30_000 },
    async () => {
      const scenario = await setupScenario();

      const result = await runScript(scenario);

      expect(result.code).toBe(0);
      expect(result.stdout).toContain("A: pulled");
      expect(result.stdout).toContain("B: skipped (dirty)");
      expect(result.stdout).toContain("refresh: pulled 1, up-to-date 0, skipped 1, failed 0");
      expect(result.stdout).not.toContain("not-a-repo");
      expect(existsSync(scenario.markerPath)).toBe(true);

      const aHead = git(scenario.aDir, ["rev-parse", "HEAD"]);
      const originHead = git(scenario.originGit, ["rev-parse", "master"]);
      expect(aHead).toBe(originHead);

      expect(existsSync(join(scenario.bDir, "scratch.txt"))).toBe(true);
    },
  );

  it(
    "with ORACLE_REFRESH_PULL=0, skips the pull phase and still runs the index command",
    { timeout: 30_000 },
    async () => {
      const scenario = await setupScenario();
      const beforeHead = git(scenario.aDir, ["rev-parse", "HEAD"]);
      const originHead = git(scenario.originGit, ["rev-parse", "master"]);
      expect(beforeHead).not.toBe(originHead);

      const result = await runScript(scenario, { ORACLE_REFRESH_PULL: "0" });

      expect(result.code).toBe(0);
      expect(result.stdout).toContain("refresh: pulled 0, up-to-date 0, skipped 0, failed 0");
      expect(existsSync(scenario.markerPath)).toBe(true);

      const afterHead = git(scenario.aDir, ["rev-parse", "HEAD"]);
      expect(afterHead).toBe(beforeHead);
      expect(afterHead).not.toBe(originHead);
    },
  );
});
