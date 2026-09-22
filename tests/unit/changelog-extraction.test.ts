import { describe, it, expect } from "vitest";
import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

// Replays scripts/extract-changelog-notes.sh -- the exact program
// .github/workflows/release.yml's "Extract changelog for this version" step
// invokes -- against the real CHANGELOG.md, so the workflow's extraction
// logic is covered by a test rather than only by hand-inspection.

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "../..");
const scriptPath = resolve(repoRoot, "scripts/extract-changelog-notes.sh");
const changelogPath = resolve(repoRoot, "CHANGELOG.md");
const pkg = JSON.parse(readFileSync(resolve(repoRoot, "package.json"), "utf8")) as {
  version: string;
};

function runExtractor(version: string) {
  return spawnSync(scriptPath, [version, changelogPath], {
    encoding: "utf8",
  });
}

describe("scripts/extract-changelog-notes.sh", () => {
  it("extracts non-empty notes for the current version, ending before the previous release heading", () => {
    const result = runExtractor(pkg.version);

    expect(result.status).toBe(0);
    expect(result.stdout.trim().length).toBeGreaterThan(0);
    // The extracted body must not run into the next "## [" heading: that
    // would mean the "found=0" reset on the next heading never fired.
    expect(result.stdout).not.toMatch(/^## \[/m);
  });

  it("fails with an explicit message when the version has no changelog heading", () => {
    const result = runExtractor("0.0.0-does-not-exist");

    expect(result.status).not.toBe(0);
    expect(result.stdout.trim()).toBe("");
    expect(result.stderr).toMatch(/no changelog notes extracted/);
  });
});
