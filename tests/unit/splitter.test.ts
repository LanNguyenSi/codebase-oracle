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
});
