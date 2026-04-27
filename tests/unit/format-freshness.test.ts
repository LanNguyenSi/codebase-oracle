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
});
