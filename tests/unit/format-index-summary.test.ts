import { describe, it, expect } from "vitest";
import { formatIndexSummary, type IndexSummary } from "../../src/ingest/runner.js";

// mcp-server.test.ts mocks formatIndexSummary, so without this test the
// actual reindex summary string (the only skip surface MCP callers see)
// would be asserted nowhere.
describe("formatIndexSummary", () => {
  const base: IndexSummary = {
    reposScanned: 2,
    filesScanned: 10,
    filesReused: 7,
    filesChanged: 2,
    filesNew: 1,
    filesPruned: 0,
    filesSkipped: 0,
    skippedFiles: [],
    chunksTotal: 40,
    chunksReused: 30,
    chunksEmbedded: 10,
    durationMs: 8700,
  };

  it("includes the skip count in the files segment", () => {
    const line = formatIndexSummary({
      ...base,
      filesSkipped: 3,
      skippedFiles: [
        { repo: "r", relativePath: "r/a.ts", reason: "too-large", sizeBytes: 600, limitBytes: 500 },
        { repo: "r", relativePath: "r/b.ts", reason: "too-large", sizeBytes: 700, limitBytes: 500 },
        { repo: "r", relativePath: "r/c.ts", reason: "read-error", message: "EACCES" },
      ],
    });
    expect(line).toContain("3 skipped");
  });

  it("keeps a stable '0 skipped' segment for zero-skip runs", () => {
    expect(formatIndexSummary(base)).toContain("0 skipped");
  });
});
