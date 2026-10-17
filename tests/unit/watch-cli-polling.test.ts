import { afterEach, beforeEach, describe, expect, it, vi, type Mock } from "vitest";
import { buildProgram } from "../../src/index.js";

// Guards the `watch --polling` CLI wiring (src/index.ts, watch command): the
// flag must reach runWatchMode as `usePolling: true`, and its absence must
// leave polling off so the default stays native fs.watch. src/watch.js is
// fully mocked, so no real watcher, store, or OS file handle is ever created;
// runWatchMode resolves to a stub `{ close }` so the action completes after
// registering its SIGINT/SIGTERM shutdown handlers, which afterEach removes
// (leaving them behind would leak listeners into other test files).

const { runWatchMode } = vi.hoisted(() => ({
  runWatchMode: vi.fn(async () => ({ close: vi.fn(async () => {}) })),
}));

vi.mock("../../src/watch.js", () => ({
  runWatchMode,
}));

async function parseWatch(argvTail: string[]): Promise<void> {
  const program = buildProgram();
  await program.parseAsync(["node", "codebase-oracle", ...argvTail]);
}

describe("oracle watch --polling CLI wiring", () => {
  let sigintBefore: (() => void)[];
  let sigtermBefore: (() => void)[];

  beforeEach(() => {
    runWatchMode.mockClear();
    sigintBefore = process
      .listeners("SIGINT")
      .map((l) => l as () => void);
    sigtermBefore = process
      .listeners("SIGTERM")
      .map((l) => l as () => void);
  });

  afterEach(() => {
    for (const listener of process.listeners("SIGINT")) {
      if (!sigintBefore.includes(listener as () => void)) {
        process.removeListener("SIGINT", listener);
      }
    }
    for (const listener of process.listeners("SIGTERM")) {
      if (!sigtermBefore.includes(listener as () => void)) {
        process.removeListener("SIGTERM", listener);
      }
    }
  });

  it("passes usePolling: true into runWatchMode when --polling is given", async () => {
    await parseWatch(["watch", "--polling"]);
    expect(runWatchMode).toHaveBeenCalledTimes(1);
    const options = (runWatchMode as Mock).mock.calls[0][1] as {
      usePolling?: boolean;
    };
    expect(options.usePolling).toBe(true);
  });

  it("passes usePolling: false (native fs.watch default) when the flag is absent", async () => {
    await parseWatch(["watch"]);
    expect(runWatchMode).toHaveBeenCalledTimes(1);
    const options = (runWatchMode as Mock).mock.calls[0][1] as {
      usePolling?: boolean;
    };
    expect(options.usePolling ?? false).toBe(false);
  });
});
