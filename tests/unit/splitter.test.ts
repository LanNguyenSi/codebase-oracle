import { describe, it, expect } from "vitest";
import { splitFile } from "../../src/ingest/splitter.js";
import type { ScannedFile } from "../../src/ingest/scanner.js";

function makeFile(overrides: Partial<ScannedFile> = {}): ScannedFile {
  return {
    absolutePath: "/tmp/test/repo/src/index.ts",
    relativePath: "repo/src/index.ts",
    repo: "repo",
    language: "ts",
    content: "export function hello() { return 'world'; }",
    contentHash: "a".repeat(64),
    ...overrides,
  };
}

describe("splitFile", () => {
  it("produces at least one chunk for a small file", async () => {
    const docs = await splitFile(makeFile());
    expect(docs.length).toBeGreaterThanOrEqual(1);
  });

  it("preserves metadata on chunks", async () => {
    const docs = await splitFile(makeFile({ repo: "my-repo", relativePath: "my-repo/src/app.ts" }));
    expect(docs[0].metadata.repo).toBe("my-repo");
    expect(docs[0].metadata.filePath).toBe("my-repo/src/app.ts");
    expect(docs[0].metadata.language).toBe("ts");
    expect(docs[0].metadata.fileHash).toBe("a".repeat(64));
  });

  it("splits large files into multiple chunks", async () => {
    const longContent = Array.from({ length: 200 }, (_, i) =>
      `export function fn${i}() { return ${i}; }`
    ).join("\n\n");

    const docs = await splitFile(makeFile({ content: longContent }));
    expect(docs.length).toBeGreaterThan(1);
  });

  it("respects chunk size limits", async () => {
    const longContent = Array.from({ length: 200 }, (_, i) =>
      `export function fn${i}() { return ${i}; }`
    ).join("\n\n");

    const docs = await splitFile(makeFile({ content: longContent }));
    for (const doc of docs) {
      // Allow some overhead from overlap
      expect(doc.pageContent.length).toBeLessThan(2000);
    }
  });

  it("handles markdown files", async () => {
    const md = "# Title\n\nIntro paragraph.\n\n## Section 1\n\nContent here.\n\n## Section 2\n\nMore content.";
    const docs = await splitFile(makeFile({ language: "md", content: md }));
    expect(docs.length).toBeGreaterThanOrEqual(1);
    expect(docs[0].metadata.language).toBe("md");
  });

  it("annotates a single-chunk file with line 1 → last source line", async () => {
    const content = "line1\nline2\nline3\nline4";
    const docs = await splitFile(makeFile({ content }));
    expect(docs).toHaveLength(1);
    expect(docs[0].metadata.lineStart).toBe(1);
    expect(docs[0].metadata.lineEnd).toBe(4);
  });

  it("locates a non-first chunk at the correct lineStart in the source", async () => {
    // Build a file where the second chunk should land predictably.
    // 100 distinct lines × 30 chars each ≈ 3000 chars total, splitter
    // (chunkSize=1500) emits at least two chunks.
    const lines = Array.from({ length: 100 }, (_, i) =>
      `export function fn${String(i).padStart(3, "0")}() { return ${i}; }`
    );
    const content = lines.join("\n");
    const docs = await splitFile(makeFile({ content }));
    expect(docs.length).toBeGreaterThanOrEqual(2);
    // The second chunk's first line must literally match the line at
    // (lineStart - 1) in the source array (1-indexed → 0-indexed conversion).
    const second = docs[1];
    const lineStart = second.metadata.lineStart as number;
    expect(lineStart).toBeGreaterThan(1);
    const firstLineOfChunk = second.pageContent.split("\n")[0];
    expect(lines[lineStart - 1]).toBe(firstLineOfChunk);
  });

  it("assigns monotonically advancing line ranges across chunks", async () => {
    const longContent = Array.from({ length: 200 }, (_, i) =>
      `export function fn${i}() { return ${i}; }`
    ).join("\n\n");

    const docs = await splitFile(makeFile({ content: longContent }));
    expect(docs.length).toBeGreaterThan(1);

    let prevStart = 0;
    for (const doc of docs) {
      const lineStart = doc.metadata.lineStart as number;
      const lineEnd = doc.metadata.lineEnd as number;
      expect(typeof lineStart).toBe("number");
      expect(typeof lineEnd).toBe("number");
      expect(lineStart).toBeGreaterThanOrEqual(1);
      expect(lineEnd).toBeGreaterThanOrEqual(lineStart);
      // Chunks come back in source order.
      expect(lineStart).toBeGreaterThanOrEqual(prevStart);
      prevStart = lineStart;
    }

    // Last chunk's lineEnd should land in the final third of the file.
    const totalLines = longContent.split("\n").length;
    const lastEnd = docs[docs.length - 1].metadata.lineEnd as number;
    expect(lastEnd).toBeGreaterThan(Math.floor(totalLines * 2 / 3));
    expect(lastEnd).toBeLessThanOrEqual(totalLines);
  });
});
