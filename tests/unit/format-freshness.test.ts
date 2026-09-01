import { describe, it, expect } from "vitest";
import {
  formatRelativeFreshness,
  formatRepoLine,
} from "../../src/format-freshness.js";

describe("formatRelativeFreshness", () => {
  const now = new Date("2026-04-27T12:00:00Z");

  it("returns 'never' for null", () => {
    expect(formatRelativeFreshness(null, now)).toBe("never");
  });

  it("returns 'never' for an unparseable timestamp", () => {
    expect(formatRelativeFreshness("not-a-date", now)).toBe("never");
  });

  it("returns 'just now' for sub-minute deltas", () => {
    const t = new Date(now.getTime() - 30 * 1000).toISOString();
    expect(formatRelativeFreshness(t, now)).toBe("just now");
  });

  it("returns 'just now' for future timestamps (clock skew)", () => {
    const t = new Date(now.getTime() + 60 * 1000).toISOString();
    expect(formatRelativeFreshness(t, now)).toBe("just now");
  });

  it("formats minutes", () => {
    const t = new Date(now.getTime() - 12 * 60 * 1000).toISOString();
    expect(formatRelativeFreshness(t, now)).toBe("12 min ago");
  });

  it("formats hours with singular/plural", () => {
    const oneHour = new Date(now.getTime() - 60 * 60 * 1000).toISOString();
    expect(formatRelativeFreshness(oneHour, now)).toBe("1 hour ago");
    const fiveHours = new Date(now.getTime() - 5 * 60 * 60 * 1000).toISOString();
    expect(formatRelativeFreshness(fiveHours, now)).toBe("5 hours ago");
  });

  it("formats days with singular/plural", () => {
    const oneDay = new Date(now.getTime() - 24 * 60 * 60 * 1000).toISOString();
    expect(formatRelativeFreshness(oneDay, now)).toBe("1 day ago");
    const threeDays = new Date(now.getTime() - 3 * 24 * 60 * 60 * 1000).toISOString();
    expect(formatRelativeFreshness(threeDays, now)).toBe("3 days ago");
  });
});

describe("formatRepoLine", () => {
  const now = new Date("2026-04-27T12:00:00Z");

  it("renders the bare line when lastIndexedAt is null (back-compat)", () => {
    expect(
      formatRepoLine(
        { repo: "scaffoldkit", chunkCount: 234, fileCount: 59, lastIndexedAt: null },
        now,
      ),
    ).toBe("- scaffoldkit — 234 chunks across 59 files");
  });

  it("appends the indexedAt suffix when a timestamp is present", () => {
    const ts = new Date(now.getTime() - 12 * 60 * 1000).toISOString();
    expect(
      formatRepoLine(
        { repo: "scaffoldkit", chunkCount: 234, fileCount: 59, lastIndexedAt: ts },
        now,
      ),
    ).toBe(`- scaffoldkit — 234 chunks across 59 files (indexed ${ts}, 12 min ago)`);
  });

  it("honors a custom prefix (e.g. CLI two-space indent)", () => {
    expect(
      formatRepoLine(
        { repo: "scaffoldkit", chunkCount: 234, fileCount: 59, lastIndexedAt: null },
        { prefix: "  " },
      ),
    ).toBe("  scaffoldkit — 234 chunks across 59 files");
  });

  it("appends nothing when skippedSizeCount/skippedErrorCount are omitted (back-compat)", () => {
    expect(
      formatRepoLine(
        { repo: "scaffoldkit", chunkCount: 234, fileCount: 59, lastIndexedAt: null },
        now,
      ),
    ).toBe("- scaffoldkit — 234 chunks across 59 files");
  });

  it("appends nothing when skippedSizeCount and skippedErrorCount are both 0", () => {
    expect(
      formatRepoLine(
        {
          repo: "scaffoldkit",
          chunkCount: 234,
          fileCount: 59,
          lastIndexedAt: null,
          skippedSizeCount: 0,
          skippedErrorCount: 0,
        },
        now,
      ),
    ).toBe("- scaffoldkit — 234 chunks across 59 files");
  });

  it("appends a skipped-file count broken down by reason when the last index run skipped files", () => {
    expect(
      formatRepoLine(
        {
          repo: "scaffoldkit",
          chunkCount: 234,
          fileCount: 59,
          lastIndexedAt: null,
          skippedSizeCount: 2,
          skippedErrorCount: 1,
        },
        now,
      ),
    ).toBe(
      "- scaffoldkit — 234 chunks across 59 files; 3 file(s) skipped in the last index run (2 too large, 1 read error)",
    );
  });

  it("singularizes 'read error' for a count of exactly 1 and renders only the too-large reason when errorCount is 0", () => {
    expect(
      formatRepoLine(
        {
          repo: "scaffoldkit",
          chunkCount: 234,
          fileCount: 59,
          lastIndexedAt: null,
          skippedSizeCount: 1,
          skippedErrorCount: 0,
        },
        now,
      ),
    ).toBe(
      "- scaffoldkit — 234 chunks across 59 files; 1 file(s) skipped in the last index run (1 too large)",
    );
    expect(
      formatRepoLine(
        {
          repo: "scaffoldkit",
          chunkCount: 234,
          fileCount: 59,
          lastIndexedAt: null,
          skippedSizeCount: 0,
          skippedErrorCount: 1,
        },
        now,
      ),
    ).toBe(
      "- scaffoldkit — 234 chunks across 59 files; 1 file(s) skipped in the last index run (1 read error)",
    );
  });

  it("lists skippedExamples after the breakdown, matching the documented shape", () => {
    expect(
      formatRepoLine(
        {
          repo: "harness",
          chunkCount: 400,
          fileCount: 80,
          lastIndexedAt: null,
          skippedSizeCount: 2,
          skippedErrorCount: 1,
          skippedExamples: ["harness/CHANGELOG.md", "harness/big.ts", "harness/locked.ts"],
        },
        now,
      ),
    ).toBe(
      "- harness — 400 chunks across 80 files; 3 file(s) skipped in the last index run "
        + "(2 too large, 1 read error; e.g. harness/CHANGELOG.md, harness/big.ts, harness/locked.ts)",
    );
  });

  it("omits the examples clause when skippedExamples is omitted or empty (back-compat)", () => {
    expect(
      formatRepoLine(
        {
          repo: "scaffoldkit",
          chunkCount: 234,
          fileCount: 59,
          lastIndexedAt: null,
          skippedSizeCount: 1,
          skippedErrorCount: 0,
          skippedExamples: [],
        },
        now,
      ),
    ).toBe(
      "- scaffoldkit — 234 chunks across 59 files; 1 file(s) skipped in the last index run (1 too large)",
    );
  });

  it("combines the indexedAt suffix and the skipped-file suffix", () => {
    const ts = new Date(now.getTime() - 12 * 60 * 1000).toISOString();
    expect(
      formatRepoLine(
        {
          repo: "scaffoldkit",
          chunkCount: 234,
          fileCount: 59,
          lastIndexedAt: ts,
          skippedSizeCount: 1,
          skippedErrorCount: 0,
        },
        now,
      ),
    ).toBe(
      `- scaffoldkit — 234 chunks across 59 files (indexed ${ts}, 12 min ago); 1 file(s) skipped in the last index run (1 too large)`,
    );
  });
});
