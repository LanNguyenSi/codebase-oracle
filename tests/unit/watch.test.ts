import { afterEach, describe, it, expect, vi } from "vitest";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, sep } from "node:path";
import type { Embeddings } from "@langchain/core/embeddings";
import {
  PendingEventMap,
  computeRepoAndRelativePath,
  runWatchMode,
  shouldIndexPath,
} from "../../src/watch.js";
import { listIndexedRepos } from "../../src/store/vector-store.js";
import type { Config } from "../../src/config.js";

function fakeEmbeddings(dimension = 8): Embeddings {
  function vec(text: string): number[] {
    const out = new Array(dimension).fill(0);
    for (let i = 0; i < text.length; i++) {
      out[i % dimension] += text.charCodeAt(i) / 1000;
    }
    const norm = Math.sqrt(out.reduce((s, v) => s + v * v, 0)) || 1;
    return out.map((v) => v / norm);
  }
  return {
    embedDocuments: async (texts: string[]) => texts.map(vec),
    embedQuery: async (t: string) => vec(t),
  } as unknown as Embeddings;
}

describe("shouldIndexPath", () => {
  const extensions = new Set([".ts", ".md", ".json"]);

  it("accepts files with an included extension", () => {
    expect(shouldIndexPath("foo.ts", extensions, true)).toBe(true);
    expect(shouldIndexPath("README.md", extensions, true)).toBe(true);
  });

  it("rejects unknown extensions", () => {
    expect(shouldIndexPath("foo.exe", extensions, true)).toBe(false);
    expect(shouldIndexPath("notes.txt", extensions, true)).toBe(false);
  });

  it("applies the JSON allowlist only when applyJsonAllowlist is true", () => {
    expect(shouldIndexPath("package.json", extensions, true)).toBe(true);
    expect(shouldIndexPath("tsconfig.json", extensions, true)).toBe(true);
    expect(shouldIndexPath("other.json", extensions, true)).toBe(false);

    // When the user provided an explicit extensions override, every .json goes through.
    expect(shouldIndexPath("other.json", extensions, false)).toBe(true);
  });
});

describe("computeRepoAndRelativePath", () => {
  const scanRoot = `${sep}tmp${sep}scan`;
  const repos = new Map<string, string>([
    [`${sep}tmp${sep}scan${sep}auth`, "auth"],
    [`${sep}tmp${sep}scan${sep}billing`, "billing"],
  ]);

  it("identifies the repo and relative path for a nested file", () => {
    const abs = `${sep}tmp${sep}scan${sep}auth${sep}src${sep}login.ts`;
    expect(computeRepoAndRelativePath(abs, scanRoot, repos)).toEqual({
      repo: "auth",
      relativePath: `auth${sep}src${sep}login.ts`,
    });
  });

  it("returns null for paths outside the scan root", () => {
    expect(
      computeRepoAndRelativePath(`${sep}etc${sep}passwd`, scanRoot, repos),
    ).toBeNull();
  });

  it("returns null for files under an unknown top-level directory", () => {
    const abs = `${sep}tmp${sep}scan${sep}unknown${sep}x.ts`;
    expect(computeRepoAndRelativePath(abs, scanRoot, repos)).toBeNull();
  });

  it("returns null for the scanRoot itself", () => {
    expect(computeRepoAndRelativePath(scanRoot, scanRoot, repos)).toBeNull();
  });
});

describe("PendingEventMap", () => {
  it("dedups multiple upserts for the same file into one entry", () => {
    const m = new PendingEventMap();
    m.recordUpsert("/abs/a", "r", "r/a.ts");
    m.recordUpsert("/abs/a", "r", "r/a.ts");
    m.recordUpsert("/abs/a", "r", "r/a.ts");
    expect(m.size()).toBe(1);
    const { files } = m.drain();
    expect(files).toHaveLength(1);
    expect(files[0].kind).toBe("upsert");
  });

  it("a delete following an upsert for the same file wins (latest event)", () => {
    const m = new PendingEventMap();
    m.recordUpsert("/abs/a", "r", "r/a.ts");
    m.recordDelete("r", "r/a.ts");
    const { files } = m.drain();
    expect(files).toHaveLength(1);
    expect(files[0].kind).toBe("delete");
  });

  it("an upsert following a delete for the same file wins (latest event)", () => {
    const m = new PendingEventMap();
    m.recordDelete("r", "r/a.ts");
    m.recordUpsert("/abs/a", "r", "r/a.ts");
    const { files } = m.drain();
    expect(files).toHaveLength(1);
    expect(files[0].kind).toBe("upsert");
  });

  it("repo-deletion drops any pending file events for that repo", () => {
    const m = new PendingEventMap();
    m.recordUpsert("/abs/a", "auth", "auth/a.ts");
    m.recordUpsert("/abs/b", "billing", "billing/b.ts");
    m.recordRepoDelete("auth");
    const { files, repos } = m.drain();
    expect(files).toHaveLength(1);
    expect(files[0].repo).toBe("billing");
    expect(repos).toEqual(["auth"]);
  });

  it("drain empties the buffers", () => {
    const m = new PendingEventMap();
    m.recordUpsert("/abs/a", "r", "r/a.ts");
    m.recordRepoDelete("r");
    m.drain();
    expect(m.size()).toBe(0);
  });
});

describe("runWatchMode integration", () => {
  const dirs: string[] = [];

  async function makeTmpDir(): Promise<string> {
    const dir = await mkdtemp(join(tmpdir(), "oracle-watch-"));
    dirs.push(dir);
    return dir;
  }

  async function setupScanRoot(): Promise<{
    scanRoot: string;
    dataDir: string;
    repoDir: string;
  }> {
    const scanRoot = await makeTmpDir();
    const dataDir = await makeTmpDir();
    const repoDir = join(scanRoot, "auth");
    await mkdir(repoDir, { recursive: true });
    await mkdir(join(repoDir, ".git"), { recursive: true });
    return { scanRoot, dataDir, repoDir };
  }

  function testConfig(scanRoot: string, dataDir: string): Config {
    return {
      scanRoot,
      dataDir,
      embeddingProvider: "openai",
      llmProvider: "auto",
      ollamaBaseUrl: "http://localhost:11434/v1",
      embeddingModel: "fake-model",
      llmModel: "fake-llm",
      vectorStoreType: "directory",
    };
  }

  afterEach(async () => {
    while (dirs.length > 0) {
      const d = dirs.pop();
      if (d) await rm(d, { recursive: true, force: true });
    }
  });

  async function waitForPending(
    watcher: { stats: () => { pending: number } },
    minCount: number,
    timeoutMs = 5000,
  ): Promise<void> {
    const start = Date.now();
    while (Date.now() - start < timeoutMs) {
      if (watcher.stats().pending >= minCount) return;
      await new Promise((r) => setTimeout(r, 40));
    }
    throw new Error(
      `timed out waiting for pending events (expected >= ${minCount}, saw ${watcher.stats().pending})`,
    );
  }

  it("embeds a newly created file within one debounce window", async () => {
    const { scanRoot, dataDir, repoDir } = await setupScanRoot();
    const config = testConfig(scanRoot, dataDir);
    const logSpy = vi.spyOn(console, "log").mockImplementation(() => {});
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

    const watcher = await runWatchMode(config, {
      embeddings: fakeEmbeddings(),
      debounceMs: 20_000, // long, so we manually flush
    });
    try {
      await writeFile(join(repoDir, "login.ts"), "export function login() { return 1; }", "utf8");
      // Give chokidar a moment to pick it up (awaitWriteFinish needs ~500ms).
      await waitForPending(watcher, 1);
      await watcher.flushOnce();

      const repos = await listIndexedRepos(config);
      expect(repos).toEqual([{ repo: "auth", chunkCount: 1, fileCount: 1 }]);
    } finally {
      await watcher.close();
      logSpy.mockRestore();
      warnSpy.mockRestore();
    }
  }, 15000);

  it("removes vectors when a file is deleted", async () => {
    const { scanRoot, dataDir, repoDir } = await setupScanRoot();
    const config = testConfig(scanRoot, dataDir);
    const logSpy = vi.spyOn(console, "log").mockImplementation(() => {});
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

    const watcher = await runWatchMode(config, {
      embeddings: fakeEmbeddings(),
      debounceMs: 20_000,
    });
    try {
      const filePath = join(repoDir, "login.ts");
      await writeFile(filePath, "export const x = 1;", "utf8");
      await waitForPending(watcher, 1);
      await watcher.flushOnce();
      expect((await listIndexedRepos(config))[0]?.chunkCount).toBeGreaterThan(0);

      await rm(filePath);
      await waitForPending(watcher, 1);
      await watcher.flushOnce();
      expect(await listIndexedRepos(config)).toEqual([]);
    } finally {
      await watcher.close();
      logSpy.mockRestore();
      warnSpy.mockRestore();
    }
  }, 15000);

  it("detects a new git repo dropped under the scan root", async () => {
    const scanRoot = await makeTmpDir();
    const dataDir = await makeTmpDir();
    const config = testConfig(scanRoot, dataDir);
    const logs: string[] = [];
    const logSpy = vi.spyOn(console, "log").mockImplementation((...args: unknown[]) => {
      logs.push(args.map(String).join(" "));
    });
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

    const watcher = await runWatchMode(config, {
      embeddings: fakeEmbeddings(),
      debounceMs: 20_000,
    });
    try {
      const newRepo = join(scanRoot, "fresh");
      await mkdir(newRepo, { recursive: true });
      await mkdir(join(newRepo, ".git"), { recursive: true });

      // Wait for the detection log line (chokidar addDir -> async stat -> log).
      const start = Date.now();
      while (Date.now() - start < 5000) {
        if (logs.some((l) => l.includes('new repo "fresh"'))) break;
        await new Promise((r) => setTimeout(r, 40));
      }
      expect(logs.some((l) => l.includes('new repo "fresh"'))).toBe(true);
    } finally {
      await watcher.close();
      logSpy.mockRestore();
      warnSpy.mockRestore();
    }
  }, 15000);

  it("collapses a save-storm into a single re-embed", async () => {
    const { scanRoot, dataDir, repoDir } = await setupScanRoot();
    const config = testConfig(scanRoot, dataDir);
    const logSpy = vi.spyOn(console, "log").mockImplementation(() => {});
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

    const fake = fakeEmbeddings();
    const embedSpy = vi.spyOn(fake, "embedDocuments");

    const watcher = await runWatchMode(config, {
      embeddings: fake,
      debounceMs: 20_000,
    });
    try {
      const filePath = join(repoDir, "storm.ts");
      // Simulate a save-storm: ten rapid writes to the same file.
      for (let i = 0; i < 10; i++) {
        await writeFile(filePath, `export const v = ${i};`, "utf8");
      }
      await waitForPending(watcher, 1);
      await watcher.flushOnce();

      expect(embedSpy).toHaveBeenCalledTimes(1);
    } finally {
      await watcher.close();
      logSpy.mockRestore();
      warnSpy.mockRestore();
    }
  }, 15000);
});
