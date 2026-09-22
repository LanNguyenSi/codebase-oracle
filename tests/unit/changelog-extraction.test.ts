import { describe, it, expect } from "vitest";
import { spawnSync } from "node:child_process";
import { readFileSync, mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve } from "node:path";

// Replays scripts/extract-changelog-notes.sh -- the exact program
// .github/workflows/release.yml's "Extract changelog for this version" step
// invokes -- against the real CHANGELOG.md, so the workflow's extraction
// logic is covered by a test rather than only by hand-inspection.

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "../..");
const scriptPath = resolve(repoRoot, "scripts/extract-changelog-notes.sh");
const changelogPath = resolve(repoRoot, "CHANGELOG.md");
const workflowPath = resolve(repoRoot, ".github/workflows/release.yml");
const pkg = JSON.parse(readFileSync(resolve(repoRoot, "package.json"), "utf8")) as {
  version: string;
};

function runExtractor(version: string, changelog: string = changelogPath) {
  return spawnSync(scriptPath, [version, changelog], {
    encoding: "utf8",
  });
}

// Writes a temp CHANGELOG.md with the given heading and body, runs the
// extractor against it, and removes the temp directory afterward.
function runExtractorAgainstFixture(version: string, body: string) {
  const tmpDir = mkdtempSync(join(tmpdir(), "changelog-extraction-"));
  try {
    const tmpChangelog = join(tmpDir, "CHANGELOG.md");
    writeFileSync(
      tmpChangelog,
      `# Changelog\n\n## [${version}] - 2026-01-01\n${body}\n## [0.0.1] - 2025-01-01\n\nOlder notes.\n`,
    );
    return runExtractor(version, tmpChangelog);
  } finally {
    rmSync(tmpDir, { recursive: true, force: true });
  }
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

  it("fails with an explicit message when the changelog file does not exist", () => {
    const result = runExtractor(pkg.version, resolve(repoRoot, "CHANGELOG.does-not-exist.md"));

    expect(result.status).not.toBe(0);
    expect(result.stdout.trim()).toBe("");
    expect(result.stderr).toMatch(/changelog file not found/);
  });

  it("fails with an explicit message when the section body is whitespace-only", () => {
    // A body made only of a line of spaces and a tab (not merely a blank
    // line) is non-empty as a raw string, so a guard that only checks
    // `[ -z "$notes" ]` would let it through. The extractor must reject it
    // the same way it rejects a truly empty section.
    const result = runExtractorAgainstFixture("9.9.9", "   \t\n");

    expect(result.status).not.toBe(0);
    expect(result.stdout.trim()).toBe("");
    expect(result.stderr).toMatch(/no changelog notes extracted/);
  });
});

describe(".github/workflows/release.yml changelog extraction step", () => {
  it("invokes scripts/extract-changelog-notes.sh from the changelog extraction step", () => {
    const workflow = readFileSync(workflowPath, "utf8");
    const stepMatch = workflow.match(
      /- name: Extract changelog for this version[\s\S]*?(?=\n {6}- name:)/,
    );
    expect(
      stepMatch,
      "could not find the 'Extract changelog for this version' step in release.yml",
    ).not.toBeNull();
    // The invocation must be a command line of the step, not a mention in a
    // comment: anchored at the start of a run-block line.
    expect(stepMatch![0]).toMatch(/^\s*scripts\/extract-changelog-notes\.sh /m);
  });
});
