import { afterEach, describe, expect, it, vi } from "vitest";
import { mkdir, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Embeddings } from "@langchain/core/embeddings";
import chokidar from "chokidar";
import type { Config } from "../../src/config.js";
import { runWatchMode } from "../../src/watch.js";

// Proves `--polling` reaches chokidar itself, not only runWatchMode: with
// chokidar mocked, runWatchMode must hand `usePolling: true` (flag on) or
// `usePolling: false` (default, native fs.watch) to the chokidar.watch call
// in src/watch.ts. The tests/unit/watch.test.ts suite covers the same
// option end-to-end with a real watcher; this file pins the exact object
// passed at the call site so a regression that drops or flips the option
// fails here first.

vi.mock("chokidar", async () => {
  const { EventEmitter } = await vi.importActual<typeof import("node:events")>(
    "node:events",
  );
  const watch = vi.fn(() => {
    const emitter = new EventEmitter();
    const fakeWatcher = {
      on: (event: string, handler: (...args: unknown[]) => void) => {
        emitter.on(event, handler);
        return fakeWatcher;
      },
      once: (event: string, handler: (...args: unknown[]) => void) => {
        emitter.once(event, handler);
        return fakeWatcher;
      },
      close: vi.fn(async () => {}),
    };
    // runWatchMode awaits "ready" synchronously after registering its
    // handlers, so a microtask lands the event once they are in place.
    queueMicrotask(() => emitter.emit("ready"));
    return fakeWatcher;
  });
  return { default: { watch } };
});

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

describe("runWatchMode forwards usePolling to chokidar.watch", () => {
  const dirs: string[] = [];

  async function setupConfig(): Promise<Config> {
    const scanRoot = await mkdtemp(join(tmpdir(), "oracle-poll-chok-"));
    dirs.push(scanRoot);
    const dataDir = await mkdtemp(join(tmpdir(), "oracle-poll-data-"));
    dirs.push(dataDir);
    const repoDir = join(scanRoot, "auth");
    await mkdir(join(repoDir, ".git"), { recursive: true });
    return {
      scanRoot,
      dataDir,
      embeddingProvider: "openai",
      llmProvider: "auto",
      ollamaBaseUrl: "http://localhost:11434/v1",
      embeddingModel: "fake-model",
      llmModel: "fake-llm",
      maxFileSizeBytes: 500_000,
      maxTextFileSizeBytes: 2_000_000,
      llmTimeoutMs: 120_000,
    };
  }

  afterEach(async () => {
    vi.mocked(chokidar.watch).mockClear();
    vi.restoreAllMocks();
    while (dirs.length > 0) {
      const d = dirs.pop();
      if (d) await rm(d, { recursive: true, force: true });
    }
  });

  async function chokidarWatchOptions(
    usePolling?: boolean,
  ): Promise<Record<string, unknown>> {
    vi.spyOn(console, "log").mockImplementation(() => {});
    const config = await setupConfig();
    const watcher = await runWatchMode(config, {
      embeddings: fakeEmbeddings(),
      ...(usePolling === undefined ? {} : { usePolling }),
    });
    await watcher.close();
    expect(chokidar.watch).toHaveBeenCalledTimes(1);
    const [path, options] = vi.mocked(chokidar.watch).mock.calls[0];
    expect(path).toBe(config.scanRoot);
    return options as unknown as Record<string, unknown>;
  }

  it("passes usePolling: true when the option is set", async () => {
    const options = await chokidarWatchOptions(true);
    expect(options.usePolling).toBe(true);
  });

  it("passes usePolling: false by default", async () => {
    const options = await chokidarWatchOptions(undefined);
    expect(options.usePolling).toBe(false);
  });
});
