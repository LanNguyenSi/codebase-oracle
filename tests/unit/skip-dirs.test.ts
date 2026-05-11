import { describe, it, expect } from "vitest";
import { DEFAULT_SKIP_DIRS, mergeSkipDirs } from "../../src/ingest/skip-dirs.js";

describe("DEFAULT_SKIP_DIRS", () => {
  it("includes vendor caches that previously polluted the index", () => {
    expect(DEFAULT_SKIP_DIRS.has(".bun")).toBe(true);
    expect(DEFAULT_SKIP_DIRS.has(".opencode-home")).toBe(true);
    expect(DEFAULT_SKIP_DIRS.has(".cache")).toBe(true);
    expect(DEFAULT_SKIP_DIRS.has(".yarn")).toBe(true);
    expect(DEFAULT_SKIP_DIRS.has(".pnpm-store")).toBe(true);
  });

  it("keeps the original defaults", () => {
    for (const name of [
      "node_modules", ".git", "dist", "build", ".next", ".turbo",
      "coverage", ".nyc_output", "__pycache__", ".venv", "vendor",
    ]) {
      expect(DEFAULT_SKIP_DIRS.has(name)).toBe(true);
    }
  });
});

describe("mergeSkipDirs", () => {
  it("returns the default set when no extras given", () => {
    expect(mergeSkipDirs(undefined)).toBe(DEFAULT_SKIP_DIRS);
    expect(mergeSkipDirs([])).toBe(DEFAULT_SKIP_DIRS);
  });

  it("appends extras without dropping defaults", () => {
    const merged = mergeSkipDirs(["custom-out", "fixtures"]);
    expect(merged.has("node_modules")).toBe(true);
    expect(merged.has(".bun")).toBe(true);
    expect(merged.has("custom-out")).toBe(true);
    expect(merged.has("fixtures")).toBe(true);
  });

  it("trims whitespace and drops empties", () => {
    const merged = mergeSkipDirs(["  spaced  ", "", "  "]);
    expect(merged.has("spaced")).toBe(true);
    expect(merged.has("")).toBe(false);
  });

  it("does not mutate the default set", () => {
    const before = DEFAULT_SKIP_DIRS.size;
    mergeSkipDirs(["one-off"]);
    expect(DEFAULT_SKIP_DIRS.size).toBe(before);
    expect(DEFAULT_SKIP_DIRS.has("one-off")).toBe(false);
  });
});
