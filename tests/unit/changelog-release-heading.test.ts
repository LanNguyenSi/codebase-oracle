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

describe("CHANGELOG.md release heading", () => {
  it("the first '## [x.y.z] - date' heading matches package.json's version", () => {
    // An empty or non-empty '## [Unreleased]' section above the first dated
    // heading is allowed: it doesn't match this pattern, so it's skipped
    // naturally rather than special-cased.
    const match = changelog.match(/^## \[(\d+\.\d+\.\d+)\] - .+$/m);
    expect(match, "no '## [x.y.z] - date' release heading found in CHANGELOG.md").not.toBeNull();
    expect(match![1]).toBe(pkg.version);
  });
});
