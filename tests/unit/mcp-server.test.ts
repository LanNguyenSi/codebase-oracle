/**
 * Tests for the createMcpServer factory extracted from mcp-server.ts.
 *
 * Heavy collaborators are vi.mock-ed so no real DB, embeddings, or network
 * runs:
 *   - createVectorStore  (src/store/vector-store.js)
 *   - createEmbeddings   (src/store/embeddings.js)
 *   - runIndex           (src/ingest/runner.js)
 *   - formatIndexSummary (src/ingest/runner.js)
 *
 * The three named behaviors under test:
 *   1. rejection-not-cached — failed store creation clears storePromise so
 *      the next call gets a fresh attempt.
 *   2. reindex mutex — a second oracle_reindex while one is in-flight returns
 *      "already running" without calling runIndex a second time.
 *   3. close-before-reindex — the cached store handle is closed (write-lock
 *      released) before runIndex opens its own connection.
 *
 * Plus one happy-path tool call (oracle_list_repos) to prove the wiring.
 *
 * Server ↔ client wiring uses InMemoryTransport.createLinkedPair() from the
 * MCP SDK — no stdio / port.
 */
import { vi, describe, it, expect, beforeEach, afterEach } from "vitest";

// vi.mock calls are hoisted above all imports by vitest's transform so these
// factories run before any module body executes.
vi.mock("../../src/store/vector-store.js", () => ({
  createVectorStore: vi.fn(),
}));

vi.mock("../../src/store/embeddings.js", () => ({
  createEmbeddings: vi.fn(() => ({
    embedDocuments: vi.fn(async () => []),
    embedQuery: vi.fn(async () => []),
  })),
}));

vi.mock("../../src/ingest/runner.js", () => ({
  runIndex: vi.fn(),
  formatIndexSummary: vi.fn(() => "Reindex complete in 0.1s.\n..."),
}));

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { createMcpServer } from "../../src/mcp-server.js";
import { createVectorStore } from "../../src/store/vector-store.js";
import { runIndex } from "../../src/ingest/runner.js";
import type { Config } from "../../src/config.js";
import type { VectorStoreWrapper } from "../../src/store/vector-store.js";
import type { IndexSummary } from "../../src/ingest/runner.js";

// Minimal config — mocks ignore config values, but we need a valid object.
const testConfig: Config = {
  dataDir: "/tmp/oracle-test-mcp",
  embeddingProvider: "openai",
  llmProvider: "auto",
  ollamaBaseUrl: "http://localhost:11434/v1",
  embeddingModel: "text-embedding-3-small",
  llmModel: "claude-sonnet-4-6",
  vectorStoreType: "directory",
};

const fakeSummary: IndexSummary = {
  reposScanned: 1,
  filesScanned: 2,
  filesReused: 1,
  filesChanged: 1,
  filesNew: 0,
  filesPruned: 0,
  chunksTotal: 4,
  chunksReused: 2,
  chunksEmbedded: 2,
  durationMs: 120,
};

function makeFakeStore(overrides: Partial<VectorStoreWrapper> = {}): VectorStoreWrapper {
  return {
    addDocuments: vi.fn(async () => {}),
    similaritySearch: vi.fn(async () => []),
    listRepos: vi.fn(() => []),
    getFileMetadata: vi.fn(() => null),
    close: vi.fn(),
    ...overrides,
  };
}

/** Connect a fresh Client to a newly-created server. Returns { client, server, cleanup }. */
async function connectClient(cfg: Config = testConfig) {
  const { server, getStore } = createMcpServer(cfg);
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  await server.connect(serverTransport);
  const client = new Client(
    { name: "test-client", version: "1.0.0" },
    { capabilities: {} },
  );
  await client.connect(clientTransport);
  const cleanup = async () => {
    try { await client.close(); } catch { /* already closed */ }
    try { await server.close(); } catch { /* already closed */ }
  };
  return { client, server, getStore, cleanup };
}

beforeEach(() => {
  vi.clearAllMocks();
});

afterEach(() => {
  vi.restoreAllMocks();
});

// ── 1. rejection-not-cached ───────────────────────────────────────────────────

describe("storePromise rejection-not-cached", () => {
  it("clears storePromise on rejection so the second call gets a fresh attempt", async () => {
    const fakeStore = makeFakeStore();
    let callCount = 0;
    vi.mocked(createVectorStore).mockImplementation(() => {
      callCount++;
      if (callCount === 1) return Promise.reject(new Error("DB connection failed"));
      return Promise.resolve(fakeStore);
    });

    const { getStore } = createMcpServer(testConfig);

    // First call rejects
    await expect(getStore()).rejects.toThrow("DB connection failed");

    // Second call resolves — proves the rejection was NOT cached.
    //
    // Mutation: if `storePromise = null` is removed from the catch block,
    // the second getStore() returns the same rejected Promise and rejects
    // again, failing this expect.
    const store = await getStore();
    expect(store).toBe(fakeStore);
    expect(callCount).toBe(2);
  });

  it("does NOT re-create the store on a subsequent successful call (caches success)", async () => {
    const fakeStore = makeFakeStore();
    vi.mocked(createVectorStore).mockResolvedValue(fakeStore);

    const { getStore } = createMcpServer(testConfig);
    const s1 = await getStore();
    const s2 = await getStore();

    expect(s1).toBe(fakeStore);
    expect(s2).toBe(fakeStore);
    // createVectorStore must only be called once (the second call reuses the cache)
    expect(vi.mocked(createVectorStore)).toHaveBeenCalledTimes(1);
  });
});

// ── 2. reindex mutex ──────────────────────────────────────────────────────────

describe("oracle_reindex mutex", () => {
  it("returns 'already running' message when a reindex is in-flight", async () => {
    let signalRunIndexStarted!: () => void;
    const runIndexStarted = new Promise<void>((res) => {
      signalRunIndexStarted = res;
    });
    let resolveRunIndex!: () => void;

    // First call to runIndex hangs until we explicitly resolve it.
    // signalRunIndexStarted is called synchronously inside the Promise
    // constructor, so by the time runIndex returns the pending Promise,
    // reindexInFlight is already true and the signal is set.
    vi.mocked(runIndex).mockImplementationOnce(
      () =>
        new Promise<IndexSummary>((res) => {
          signalRunIndexStarted();
          resolveRunIndex = () => res(fakeSummary);
        }),
    );

    const { client, cleanup } = await connectClient();

    try {
      // Fire first oracle_reindex (will hang at runIndex).
      // Because InMemoryTransport.send calls onmessage synchronously, the
      // server handler runs up to `await runIndex(cfg)` BEFORE client.callTool
      // returns — so reindexInFlight is already true and the signal is
      // already set by the time we reach the next line.
      const firstCallP = client.callTool({ name: "oracle_reindex", arguments: {} });

      // Wait for runIndex to be called (belt-and-suspenders: handles any
      // async scheduling model the MCP SDK may use).
      await runIndexStarted;

      // Second call while first is in-flight → must get "already running".
      //
      // Mutation: removing `if (reindexInFlight) { return "already running" }`
      // causes the second call to start a real runIndex call, which returns
      // a different message (the summary from formatIndexSummary), failing
      // the toContain assertion below.
      const secondResult = await client.callTool({ name: "oracle_reindex", arguments: {} });
      const text = (secondResult.content as Array<{ type: string; text: string }>)[0]?.text ?? "";
      expect(text).toContain("already running");
      expect(text).toContain("Wait for it to finish");

      // Release first call so the transport can be cleanly closed.
      resolveRunIndex();
      await firstCallP;
    } finally {
      await cleanup();
    }
  });
});

// ── 3. close-before-reindex ───────────────────────────────────────────────────

describe("oracle_reindex close-before-reindex", () => {
  it("calls store.close() before runIndex to release the write lock", async () => {
    const callOrder: string[] = [];

    const closeSpy = vi.fn(() => {
      callOrder.push("close");
    });
    const fakeStore = makeFakeStore({ close: closeSpy });
    vi.mocked(createVectorStore).mockResolvedValue(fakeStore);

    vi.mocked(runIndex).mockImplementation(async () => {
      callOrder.push("runIndex");
      return fakeSummary;
    });

    // connectClient creates the server and connects it to ONE transport;
    // we reuse that same client for the oracle_reindex call below.
    const { getStore, client, cleanup } = await connectClient();

    try {
      // Prime the store cache via the white-box getStore() seam.
      // After this, storePromise is set and the oracle_reindex handler
      // will call handle.close() before running runIndex.
      await getStore();

      await client.callTool({ name: "oracle_reindex", arguments: {} });

      // close() must be called before runIndex()
      //
      // Mutation: swapping the two lines (calling runIndex before close) would
      // produce callOrder = ["runIndex", "close"], failing the indexOf check.
      expect(callOrder.indexOf("close")).toBeGreaterThanOrEqual(0);
      expect(callOrder.indexOf("runIndex")).toBeGreaterThanOrEqual(0);
      expect(callOrder.indexOf("close")).toBeLessThan(callOrder.indexOf("runIndex"));
    } finally {
      await cleanup();
    }
  });
});

// ── 4. happy-path oracle_list_repos ──────────────────────────────────────────

describe("oracle_list_repos happy path", () => {
  it("returns the empty-index message when the store has no repos", async () => {
    vi.mocked(createVectorStore).mockResolvedValue(makeFakeStore({ listRepos: vi.fn(() => []) }));

    const { client, cleanup } = await connectClient();

    try {
      const result = await client.callTool({ name: "oracle_list_repos", arguments: {} });
      const text = (result.content as Array<{ type: string; text: string }>)[0]?.text ?? "";
      expect(text).toContain("No repos in the index yet");
    } finally {
      await cleanup();
    }
  });
});
