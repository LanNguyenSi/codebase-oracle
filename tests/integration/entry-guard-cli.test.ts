import { afterEach, describe, it, expect } from "vitest";
import { mkdtemp, rm, symlink, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

// Guards src/index.ts's `isMainModule` entry check, which decides whether
// the CLI actually runs (vs. just exporting buildProgram() for imports like
// tests/unit/readme-cli-flags.test.ts). `process.argv[1]` is path.resolve'd
// but keeps any symlink segment, while `import.meta.url` is realpath'd by
// Node, so a strict `===` compare silently fails whenever the entry file is
// reached through a symlink: the npm bin shim
// (node_modules/.bin/codebase-oracle), `npx`, a global install, and
// `claude mcp add codebase-oracle -- codebase-oracle mcp` (docs/mcp.md) all
// go through exactly this path. The failure mode is silent: the process
// exits 0 with no output at all, so this test asserts on stdout content,
// not just exit status.

const repoRoot = fileURLToPath(new URL("../..", import.meta.url));
const indexEntry = join(repoRoot, "src", "index.ts");

const tmpDirs: string[] = [];
async function makeTmpDir(): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), "oracle-entry-guard-"));
  tmpDirs.push(dir);
  return dir;
}

afterEach(async () => {
  while (tmpDirs.length > 0) {
    const dir = tmpDirs.pop();
    if (dir) await rm(dir, { recursive: true, force: true });
  }
});

describe("oracle CLI entry guard, run through a symlink", () => {
  it(
    "prints --version output when invoked via a symlink to src/index.ts",
    { timeout: 30_000 },
    async () => {
      const pkg = JSON.parse(
        await readFile(join(repoRoot, "package.json"), "utf8"),
      ) as { version: string };

      const dir = await makeTmpDir();
      const entrySymlink = join(dir, "codebase-oracle-entry.ts");
      // Mirrors how the real CLI is actually invoked in production: the npm
      // bin shim, `npx`, and a global install all reach src's compiled
      // entry through a symlink rather than a direct path.
      await symlink(indexEntry, entrySymlink);

      const result = spawnSync("npx", ["tsx", entrySymlink, "--version"], {
        encoding: "utf8",
        cwd: repoRoot,
      });

      // The bug's signature is a *silent* no-op: exit 0 with empty stdout.
      // Asserting only on exit code would miss it, so assert on the actual
      // printed output.
      expect(
        result.stdout.trim().length,
        `expected non-empty stdout, got: stdout=${JSON.stringify(result.stdout)} stderr=${JSON.stringify(result.stderr)}`,
      ).toBeGreaterThan(0);
      expect(result.stdout).toContain(pkg.version);
      expect(result.status).toBe(0);
    },
  );
});
