import { afterEach, describe, it, expect } from "vitest";
import { existsSync } from "node:fs";
import { mkdtemp, rm, writeFile, mkdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { execFile, execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";

// Drives the real scripts/oracle-refresh.sh over throwaway local git repos:
// a bare "origin" and clones covering every classification the pull loop
// makes (pulled, up-to-date, dirty, detached, no-upstream, failed) plus a
// plain non-repo directory that must be ignored. Also exercises the exit
// status passthrough and the ORACLE_SCAN_ROOT .env fallback in isolation
// from a real ~/.codebase-oracle or the checkout's own .env.

const repoRoot = fileURLToPath(new URL("../..", import.meta.url));
const scriptPath = join(repoRoot, "scripts", "oracle-refresh.sh");

// Isolate every git invocation (script's and the test's setup) from the
// operator's real git config / any ambient GIT_* pointing elsewhere.
// GIT_DIR/GIT_WORK_TREE are cleared via `undefined` (fully unset), not `""`:
// git treats an empty-but-present GIT_WORK_TREE as an explicit override and
// refuses to run without a matching GIT_DIR, which is the opposite of the
// intent here (make sure neither is inherited from the ambient shell).
const GIT_ENV: Record<string, string | undefined> = {
  GIT_CONFIG_GLOBAL: "/dev/null",
  GIT_CONFIG_SYSTEM: "/dev/null",
  GIT_DIR: undefined,
  GIT_WORK_TREE: undefined,
};

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
  return execFileSync("git", args, {
    cwd: dir,
    encoding: "utf8",
    env: { ...process.env, ...GIT_ENV },
  }).trim();
}

function gitQuiet(dir: string, args: string[]): void {
  execFileSync("git", args, {
    cwd: dir,
    encoding: "utf8",
    env: { ...process.env, ...GIT_ENV },
  });
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
  cDir: string;
  dDir: string;
  eDir: string;
  fDir: string;
  markerPath: string;
}

async function setupScenario(): Promise<Scenario> {
  const workDir = await makeTmpDir("oracle-refresh-work-");
  const scanRoot = await makeTmpDir("oracle-refresh-scan-");
  const dataDir = await makeTmpDir("oracle-refresh-data-");

  const originGit = join(workDir, "origin.git");
  execFileSync("git", ["init", "--quiet", "--bare", originGit], {
    env: { ...process.env, ...GIT_ENV },
  });
  // Pin the bare repo's default branch so clones don't depend on the
  // machine's global init.defaultBranch setting.
  gitQuiet(originGit, ["symbolic-ref", "HEAD", "refs/heads/master"]);

  const seedDir = join(workDir, "seed");
  execFileSync("git", ["clone", "--quiet", originGit, seedDir], {
    env: { ...process.env, ...GIT_ENV },
  });
  configureIdentity(seedDir);
  await writeFile(join(seedDir, "file1.txt"), "one\n", "utf8");
  git(seedDir, ["add", "file1.txt"]);
  git(seedDir, ["commit", "-q", "-m", "commit 1"]);
  git(seedDir, ["push", "-q", "-u", "origin", "master"]);

  const clone = (name: string): string => {
    const dir = join(scanRoot, name);
    execFileSync("git", ["clone", "--quiet", originGit, dir], {
      env: { ...process.env, ...GIT_ENV },
    });
    configureIdentity(dir);
    return dir;
  };

  // A: clean, tracked, behind -> gets pulled.
  const aDir = clone("A");
  // B: dirty via an untracked file -> skipped regardless of pull outcome.
  const bDir = clone("B");
  // C: detached HEAD -> skipped.
  const cDir = clone("C");
  // D: local-only branch, no upstream -> skipped.
  const dDir = clone("D");
  // E: local commit diverges from origin's advance -> pull fails.
  const eDir = clone("E");

  // Advance origin beyond A/B/C/D/E's current HEAD.
  await writeFile(join(seedDir, "file2.txt"), "two\n", "utf8");
  git(seedDir, ["add", "file2.txt"]);
  git(seedDir, ["commit", "-q", "-m", "commit 2"]);
  git(seedDir, ["push", "-q", "origin", "master"]);

  // F: cloned only after origin's second commit -> already up to date.
  const fDir = clone("F");

  // B is dirty via an untracked file. Untracked (not modified-tracked) so a
  // pull would not itself conflict; the script must still skip it purely on
  // the "checkout is dirty" policy.
  await writeFile(join(bDir, "scratch.txt"), "scratch\n", "utf8");

  // C: detach HEAD at its current commit.
  git(cDir, ["checkout", "--quiet", "--detach", "HEAD"]);

  // D: switch to a new local branch with no upstream tracking.
  git(dDir, ["checkout", "--quiet", "-b", "local-only"]);

  // E: commit locally without pushing so origin's advance and E's local
  // commit diverge; `git pull --ff-only` must fail (non-fast-forward).
  await writeFile(join(eDir, "local-only.txt"), "local\n", "utf8");
  git(eDir, ["add", "local-only.txt"]);
  git(eDir, ["commit", "-q", "-m", "local divergent commit"]);

  // A plain directory with no .git must be ignored entirely.
  const notARepo = join(scanRoot, "not-a-repo");
  await mkdir(notARepo, { recursive: true });
  await writeFile(join(notARepo, "readme.txt"), "not a repo\n", "utf8");

  const markerPath = join(workDir, "indexed");

  return { scanRoot, dataDir, originGit, aDir, bDir, cDir, dDir, eDir, fDir, markerPath };
}

function runScriptWithEnv(
  env: Record<string, string | undefined>,
): Promise<{ stdout: string; stderr: string; code: number }> {
  return new Promise((resolvePromise) => {
    execFile(
      "bash",
      [scriptPath],
      {
        cwd: repoRoot,
        env,
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

function runScript(
  scenario: Scenario,
  extraEnv: Record<string, string> = {},
): Promise<{ stdout: string; stderr: string; code: number }> {
  return runScriptWithEnv({
    ...process.env,
    ...GIT_ENV,
    ORACLE_SCAN_ROOT: scenario.scanRoot,
    ORACLE_DATA_DIR: scenario.dataDir,
    ORACLE_REFRESH_INDEX_CMD: `touch "${scenario.markerPath}"`,
    ORACLE_REFRESH_PULL: "1",
    ...extraEnv,
  });
}

describe("scripts/oracle-refresh.sh", () => {
  it(
    "classifies every scenario repo, reports a pull failure's reason, leaves untouched HEADs alone, and runs the index command",
    { timeout: 30_000 },
    async () => {
      const scenario = await setupScenario();

      const cHeadBefore = git(scenario.cDir, ["rev-parse", "HEAD"]);
      const dHeadBefore = git(scenario.dDir, ["rev-parse", "HEAD"]);
      const eHeadBefore = git(scenario.eDir, ["rev-parse", "HEAD"]);
      const fHeadBefore = git(scenario.fDir, ["rev-parse", "HEAD"]);

      const result = await runScript(scenario);

      expect(result.code).toBe(0);
      expect(result.stdout).toContain("A: pulled");
      expect(result.stdout).toContain("B: skipped (dirty)");
      expect(result.stdout).toContain("C: skipped (detached)");
      expect(result.stdout).toContain("D: skipped (no-upstream)");
      expect(result.stdout).toContain("E: failed");
      expect(result.stdout).toContain("F: up-to-date");
      expect(result.stdout).toContain("refresh: pulled 1, up-to-date 1, skipped 3, failed 1");
      expect(result.stdout).not.toContain("not-a-repo");
      expect(existsSync(scenario.markerPath)).toBe(true);

      // The failed pull's reason line follows directly after "E: failed".
      const lines = result.stdout.split("\n");
      const failedIndex = lines.findIndex((l) => l.startsWith("E: failed"));
      expect(failedIndex).toBeGreaterThanOrEqual(0);
      expect(lines[failedIndex + 1]).toMatch(/^\s*reason: .+/);

      const aHead = git(scenario.aDir, ["rev-parse", "HEAD"]);
      const originHead = git(scenario.originGit, ["rev-parse", "master"]);
      expect(aHead).toBe(originHead);

      expect(existsSync(join(scenario.bDir, "scratch.txt"))).toBe(true);
      expect(git(scenario.cDir, ["rev-parse", "HEAD"])).toBe(cHeadBefore);
      expect(git(scenario.dDir, ["rev-parse", "HEAD"])).toBe(dHeadBefore);
      expect(git(scenario.eDir, ["rev-parse", "HEAD"])).toBe(eHeadBefore);
      expect(git(scenario.fDir, ["rev-parse", "HEAD"])).toBe(fHeadBefore);
      expect(fHeadBefore).toBe(originHead);
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

  it(
    "propagates the index command's own exit status when the pull phase is skipped",
    { timeout: 30_000 },
    async () => {
      const scenario = await setupScenario();

      const result = await runScript(scenario, {
        ORACLE_REFRESH_PULL: "0",
        ORACLE_REFRESH_INDEX_CMD: "exit 42",
      });

      expect(result.code).toBe(42);
    },
  );

  it(
    "ORACLE_SCAN_ROOT env var takes precedence over a differing .env value",
    { timeout: 30_000 },
    async () => {
      const scenario = await setupScenario();
      const envDir = await makeTmpDir("oracle-refresh-envfile-");
      const envFile = join(envDir, ".env");
      await writeFile(envFile, "ORACLE_SCAN_ROOT=/nonexistent/should-not-be-used\n", "utf8");

      const result = await runScriptWithEnv({
        ...process.env,
        ...GIT_ENV,
        ORACLE_SCAN_ROOT: scenario.scanRoot,
        ORACLE_REFRESH_ENV_FILE: envFile,
        ORACLE_REFRESH_PULL: "0",
        ORACLE_REFRESH_INDEX_CMD: `touch "${scenario.markerPath}"`,
      });

      expect(result.code).toBe(0);
      expect(result.stderr).not.toContain("is not a directory");
      expect(existsSync(scenario.markerPath)).toBe(true);
    },
  );

  it(
    "falls back to a quoted ORACLE_SCAN_ROOT= line with spaced '=' and a CRLF ending in .env",
    { timeout: 30_000 },
    async () => {
      const scenario = await setupScenario();
      const envDir = await makeTmpDir("oracle-refresh-envfile-");
      const envFile = join(envDir, ".env");
      await writeFile(envFile, `ORACLE_SCAN_ROOT = '${scenario.scanRoot}' \r\n`, "utf8");

      const result = await runScriptWithEnv({
        ...process.env,
        ...GIT_ENV,
        ORACLE_SCAN_ROOT: undefined,
        ORACLE_REFRESH_ENV_FILE: envFile,
        ORACLE_REFRESH_PULL: "0",
        ORACLE_REFRESH_INDEX_CMD: `touch "${scenario.markerPath}"`,
      });

      expect(result.code).toBe(0);
      expect(result.stderr).not.toContain("is not a directory");
      expect(result.stderr).not.toContain("is not set");
      expect(existsSync(scenario.markerPath)).toBe(true);
    },
  );

  it(
    "aborts with a message naming ORACLE_SCAN_ROOT when neither the env var nor .env supplies one",
    { timeout: 30_000 },
    async () => {
      const envDir = await makeTmpDir("oracle-refresh-envfile-");
      const envFile = join(envDir, ".env");
      // No ORACLE_SCAN_ROOT= line at all.
      await writeFile(envFile, "SOME_OTHER_VAR=1\n", "utf8");

      const result = await runScriptWithEnv({
        ...process.env,
        ...GIT_ENV,
        ORACLE_SCAN_ROOT: undefined,
        ORACLE_REFRESH_ENV_FILE: envFile,
      });

      expect(result.code).not.toBe(0);
      expect(result.stderr).toContain("ORACLE_SCAN_ROOT");
    },
  );
});
