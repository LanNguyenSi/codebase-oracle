import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "../..");
const pkg = JSON.parse(readFileSync(resolve(repoRoot, "package.json"), "utf8")) as {
  version: string;
};
const changelog = readFileSync(resolve(repoRoot, "CHANGELOG.md"), "utf8");

// A semver core plus an optional prerelease suffix (e.g. "0.13.0-rc.1"), so a
// tagged prerelease is not silently rejected by this guard while the release
// workflow itself (which matches the version as a plain string, not through
// this regex) accepts it.
const HEADING_RE = /^## \[(\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?)\] - .+$/m;

/** The first "## [x.y.z...] - date" heading's version, or undefined. */
function firstHeadingVersion(markdown: string): string | undefined {
  return markdown.match(HEADING_RE)?.[1];
}

/**
 * Whether the section under the first dated heading has non-whitespace
 * content before the next "## [" heading (or end of file). An empty or
 * whitespace-only section would still slip past a release: this is what
 * catches it at CI time, before npm publish runs.
 */
function firstSectionHasNonWhitespaceBody(markdown: string): boolean {
  const match = markdown.match(HEADING_RE);
  if (!match || match.index === undefined) return false;
  const headingEnd = match.index + match[0].length;
  const rest = markdown.slice(headingEnd);
  const nextHeadingMatch = rest.match(/^## \[/m);
  const body = nextHeadingMatch ? rest.slice(0, nextHeadingMatch.index) : rest;
  return /\S/.test(body);
}

describe("CHANGELOG.md release heading", () => {
  it("the first '## [x.y.z] - date' heading matches package.json's version", () => {
    // An empty or non-empty '## [Unreleased]' section above the first dated
    // heading is allowed: it doesn't match this pattern, so it's skipped
    // naturally rather than special-cased.
    const version = firstHeadingVersion(changelog);
    expect(version, "no '## [x.y.z] - date' release heading found in CHANGELOG.md").not.toBeUndefined();
    expect(version).toBe(pkg.version);
  });

  it("the matched dated section has non-whitespace body content before the next heading", () => {
    expect(
      firstSectionHasNonWhitespaceBody(changelog),
      "the first dated CHANGELOG.md section is empty or whitespace-only",
    ).toBe(true);
  });

  it("matches a prerelease heading against a prerelease package.json version (fixture)", () => {
    const fixture =
      "## [Unreleased]\n\n## [0.13.0-rc.1] - 2026-02-01\n\nRelease-candidate notes.\n\n## [0.12.0] - 2026-01-01\n";
    expect(firstHeadingVersion(fixture)).toBe("0.13.0-rc.1");
  });

  it("does not match a prerelease package.json version against a bare-core heading (fixture)", () => {
    const fixture = "## [Unreleased]\n\n## [0.13.0] - 2026-02-01\n\nNotes.\n\n## [0.12.0] - 2026-01-01\n";
    const fixturePkgVersion = "0.13.0-rc.1";
    expect(firstHeadingVersion(fixture)).not.toBe(fixturePkgVersion);
  });

  it("flags an empty-body dated section via a fixture (would fail in ci before npm publish)", () => {
    const fixture = "## [Unreleased]\n\n## [1.2.3] - 2026-01-01\n\n## [1.2.2] - 2025-12-01\n\nOlder notes.\n";
    expect(firstSectionHasNonWhitespaceBody(fixture)).toBe(false);
  });
});
