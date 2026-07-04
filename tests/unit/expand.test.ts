import { describe, it, expect } from "vitest";
import { mkdtemp, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { expandFile, formatExpandResult } from "../../src/expand.js";
import type { VectorStoreWrapper } from "../../src/store/vector-store.js";

interface FakeStoreOptions {
  files?: Record<string, Record<string, unknown>>; // key: "repo::path" → metadata
}

function fakeStore(opts: FakeStoreOptions = {}): VectorStoreWrapper {
  const files = opts.files ?? {};
  return {
    addDocuments: async () => {},
    similaritySearch: async () => [],
    listRepos: () => [],
    getFileMetadata: (repo, path) => files[`${repo}::${path}`] ?? null,
    getFirstChunkByFile: () => null,
    close: () => {},
  };
}

async function makeTmpDir(): Promise<string> {
  return mkdtemp(join(tmpdir(), "oracle-expand-"));
}

describe("expandFile", () => {
  it("returns not_indexed when no chunks exist for the file", async () => {
    const result = await expandFile(fakeStore(), {
      repo: "ghost",
      path: "ghost/x.ts",
    });
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.reason).toBe("not_indexed");
      expect(result.message).toContain("ghost/x.ts");
    }
  });

  it("returns no_absolute_path when metadata is missing absolutePath", async () => {
    const store = fakeStore({
      files: { "r::r/a.ts": { repo: "r", filePath: "r/a.ts" } },
    });
    const result = await expandFile(store, { repo: "r", path: "r/a.ts" });
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.reason).toBe("no_absolute_path");
  });

  it("returns file_missing when the indexed absolutePath has been deleted", async () => {
    const store = fakeStore({
      files: {
        "r::r/a.ts": { absolutePath: "/nonexistent-/oracle-test/missing.ts" },
      },
    });
    const result = await expandFile(store, { repo: "r", path: "r/a.ts" });
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.reason).toBe("file_missing");
  });

  it("reads a centered window around the requested line", async () => {
    const dir = await makeTmpDir();
    try {
      const abs = join(dir, "code.ts");
      const lines = Array.from({ length: 100 }, (_, i) => `line${i + 1}`);
      await writeFile(abs, lines.join("\n"), "utf8");

      const store = fakeStore({ files: { "r::r/code.ts": { absolutePath: abs } } });
      const result = await expandFile(store, {
        repo: "r",
        path: "r/code.ts",
        line: 50,
        window: 10,
      });
      expect(result.ok).toBe(true);
      if (result.ok) {
        // Symmetric window of 10 around line 50: half=5, so lineStart=45, lineEnd=54.
        expect(result.lineStart).toBe(45);
        expect(result.lineEnd).toBe(54);
        expect(result.totalLines).toBe(100);
        expect(result.text).toContain("45→line45");
        expect(result.text).toContain("54→line54");
        // Window respected.
        expect(result.text).not.toContain("→line44");
        expect(result.text).not.toContain("→line55");
      }
    } finally {
      await rm(dir, { recursive: true });
    }
  });

  it("clamps the window at the start of the file", async () => {
    const dir = await makeTmpDir();
    try {
      const abs = join(dir, "code.ts");
      await writeFile(abs, "a\nb\nc\nd\ne\nf\ng", "utf8");
      const store = fakeStore({ files: { "r::r/code.ts": { absolutePath: abs } } });
      const result = await expandFile(store, {
        repo: "r",
        path: "r/code.ts",
        line: 1,
        window: 10,
      });
      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.lineStart).toBe(1);
        expect(result.lineEnd).toBe(7);
      }
    } finally {
      await rm(dir, { recursive: true });
    }
  });

  it("clamps the window at the end of the file", async () => {
    const dir = await makeTmpDir();
    try {
      const abs = join(dir, "code.ts");
      const lines = Array.from({ length: 50 }, (_, i) => `line${i + 1}`);
      await writeFile(abs, lines.join("\n"), "utf8");
      const store = fakeStore({ files: { "r::r/code.ts": { absolutePath: abs } } });
      const result = await expandFile(store, {
        repo: "r",
        path: "r/code.ts",
        line: 49,
        window: 10,
      });
      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.lineEnd).toBe(50);
      }
    } finally {
      await rm(dir, { recursive: true });
    }
  });

  it("strips trailing CR from CRLF-encoded files", async () => {
    const dir = await makeTmpDir();
    try {
      const abs = join(dir, "windows.ts");
      // Write CRLF-encoded content; split("\n") leaves \r on each line.
      await writeFile(abs, "alpha\r\nbeta\r\ngamma\r\n", "utf8");
      const store = fakeStore({ files: { "r::r/win.ts": { absolutePath: abs } } });
      const result = await expandFile(store, {
        repo: "r",
        path: "r/win.ts",
        line: 2,
        window: 3,
      });
      expect(result.ok).toBe(true);
      if (result.ok) {
        // No literal \r should leak into the rendered text.
        expect(result.text).not.toContain("\r");
        expect(result.text).toContain("→alpha");
        expect(result.text).toContain("→beta");
        expect(result.text).toContain("→gamma");
      }
    } finally {
      await rm(dir, { recursive: true });
    }
  });

  it("caps the window at 200 lines", async () => {
    const dir = await makeTmpDir();
    try {
      const abs = join(dir, "huge.ts");
      const lines = Array.from({ length: 1000 }, (_, i) => `line${i + 1}`);
      await writeFile(abs, lines.join("\n"), "utf8");
      const store = fakeStore({ files: { "r::r/huge.ts": { absolutePath: abs } } });
      const result = await expandFile(store, {
        repo: "r",
        path: "r/huge.ts",
        line: 500,
        window: 999, // way over the cap
      });
      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.lineEnd - result.lineStart + 1).toBe(200);
      }
    } finally {
      await rm(dir, { recursive: true });
    }
  });
});

describe("formatExpandResult", () => {
  it("renders a header matching the oracle_search convention", () => {
    const text = formatExpandResult({
      ok: true,
      repo: "r",
      path: "r/a.ts",
      lineStart: 5,
      lineEnd: 7,
      totalLines: 100,
      text: " 5→fn(\n 6→  return 1;\n 7→}",
    });
    expect(text).toContain("r/a.ts:5-7 (r, 100 lines total)");
    expect(text).toContain("5→fn(");
    expect(text).toContain("7→}");
  });

  it("renders a friendly error for failure cases", () => {
    const text = formatExpandResult({
      ok: false,
      reason: "not_indexed",
      message: "no chunks for x",
    });
    expect(text).toBe("oracle_expand: no chunks for x");
  });
});
