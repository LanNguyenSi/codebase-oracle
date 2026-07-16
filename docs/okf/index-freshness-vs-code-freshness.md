---
type: runbook
title: Index-data freshness vs server-code freshness
description: Two independent staleness axes in codebase-oracle — reindexed store data is visible to a running MCP server without restart (WAL), but changed oracle source needs npm run build plus an MCP-client reconnect; verify each separately.
tags: [runbook, mcp, indexing, freshness, dist]
timestamp: 2026-07-16T02:36:27Z
sources:
  - package.json
  - src/store/sqlite-store.ts
  - src/mcp-server.ts
  - src/version.ts
  - src/config.ts
  - src/index.ts
  - docs/architecture.md
---

# Index-data freshness vs server-code freshness

You changed something and the running oracle does not reflect it. The common
folklore — "a running MCP server keeps pre-change state until you reconnect" —
collapses **two independent things** into one. Diagnose which one is stale, then
apply the matching fix. They have different fixes and different verifications.

- **Index DATA** = the content of the SQLite store (`store.db`): chunks,
  embeddings, per-repo freshness. Changed by (re)indexing.
- **Server CODE** = the codebase-oracle binary the MCP client executes. Changed
  by editing `src/` and rebuilding.

## Decision table: you changed X → Y is stale → do Z

| You changed | What is stale | Fix | Restart the MCP server? |
|---|---|---|---|
| Repo content you want searchable | Index DATA | `npm run index` (CLI) **or** `oracle_reindex` (MCP tool) | **No** — a running server sees it on its next query |
| codebase-oracle's own `src/` | Server CODE (only if client runs the compiled `dist` bin) | `npm run build`, then reconnect the MCP client | **Yes** — client must re-spawn the process |
| `.env` / `ORACLE_*` config | Neither of the above cleanly — see "Config" below | edit `.env`, then reconnect the MCP client | **Yes** — env is parsed once at process start |

## 1) Index DATA freshness — no restart needed

Verified cross-process visibility:

- The store is a single SQLite file opened with WAL:
  `db.pragma("journal_mode = WAL")` plus `synchronous = NORMAL` and
  `busy_timeout = 5000` (`src/store/sqlite-store.ts:178-184`). WAL lets readers
  and a separate writer share one store; the header comment states exactly this
  (`src/store/sqlite-store.ts:180-184`).
- Every read runs a **fresh prepared statement** each call — `listRepos.all()`,
  the `similaritySearch` `db.prepare(sql).all(...)`
  (`src/store/sqlite-store.ts:398, 436, 459`). There is no long-lived read
  transaction and no JS-level result cache, so each query observes the latest
  committed state, including writes committed by *another* process (a CLI
  `npm run index`, `npm run watch`, or a systemd reindex).
- `docs/architecture.md:46-49` asserts the same guarantee: WAL store, "Writes
  and reads can safely happen from different processes," and "A running stdio or
  HTTP MCP server sees `npm run watch` writes on its next query without
  restarting." **The code supports this claim.**

Practical consequence: after reindexing, you do **not** restart the MCP server.
The next `oracle_search` / `oracle_query` / `oracle_list_repos` call re-reads the
store. `oracle_reindex` even closes and drops the cached store handle so the
indexer does not contend for the write lock; the next tool call re-opens it
(`src/mcp-server.ts:246-260`).

Note on `writeEpoch`: the store exposes a `bumpWriteEpoch()` / `getWriteEpoch()`
heartbeat bumped on every mutation (`src/store/sqlite-store.ts:120-122,
501, 525, 676-685`). It is written but **no reader in `src/` currently consumes
`getWriteEpoch()` to invalidate a cache** — the freshness guarantee rests on WAL
plus per-statement reads, not on epoch polling. Do not rely on the epoch as the
mechanism; it is a hook, not the load-bearing part.

### Verify new chunks are visible

1. Reindex via the **CLI** to isolate the ingest from the server:
   `npm run index` (which is `tsx src/index.ts index`, `package.json:23`).
2. From the still-running MCP session call `oracle_list_repos` and check the
   per-repo chunk/file counts and the indexed timestamp (rendered from
   `last_indexed_at`; `src/store/sqlite-store.ts:268-277`,
   `src/format-freshness.ts`). Or call `oracle_search` for a string you know is
   only in the new content. A hit without any restart confirms DATA freshness.

## 2) Server CODE freshness — build + reconnect, and it depends how the client launches it

The published/`bin` entry is a **compiled artifact**:
`"bin": { "codebase-oracle": "./dist/index.js" }` (`package.json:6-8`). Editing
`src/` does nothing for a process running that binary until you rebuild:
`"build": "tsc"` (`package.json:21`). Even after rebuild, a *running* server
still executes the already-`exec`'d old binary — the MCP client must
reconnect/restart the server process to `exec` the new one.

Crucial nuance — **how your MCP client launches the server decides whether a
build is even needed**:

- If the client runs the compiled bin (`node dist/index.js mcp`, or the
  installed `codebase-oracle mcp`), then: edit `src/` → **`npm run build`** →
  reconnect.
- If the client runs `npm run mcp`, that is `tsx src/mcp-server.ts`
  (`package.json:26`) — it transpiles **source** on each spawn. No `npm run
  build` is needed; a fresh spawn is already code-fresh. But the server is still
  a long-lived process, so you **still must reconnect** for edits to take
  effect. (`npm run dev` / `index` / `query` / `watch` are likewise `tsx
  src/...`, `package.json:22-27`.)

So the folklore is only half-right: a running server never hot-reloads its own
code, but a *newly spawned* server is code-fresh only on the `tsx` path, and
needs a build on the `dist` path.

### Verify the running binary is the new one

- The version string flows from `package.json` through `src/version.ts`
  (`VERSION = pkg.version`, reading `../package.json`) into the MCP handshake
  (`new McpServer({ name: "codebase-oracle", version: VERSION })`,
  `src/mcp-server.ts:19, 37-41`). If you bumped the version, the reconnected
  client's server-info version proves the new process is live. Note: `VERSION`
  tracks `package.json`, **not** arbitrary `src/` edits, so a same-version code
  change is not distinguishable this way — for those, exercise the changed
  behavior directly after reconnecting.
- On the `dist` path, if behavior did not change, confirm you actually rebuilt:
  `dist/` is only regenerated by `npm run build` (`prepublishOnly` also runs it,
  `package.json:30`). A stale `dist` is the classic "I edited src and nothing
  happened" trap.

## Config (`.env` / `ORACLE_*`) — parsed once at process start

Config is **not** re-read per call. `src/mcp-server.ts:21-23` runs
`loadEnvFromFile()` and `const config = loadConfig()` at **module load**, once.
`loadConfig()` reads `process.env.ORACLE_*` (`src/config.ts:97-134`), and
`loadEnvFromFile()` only sets vars that are not already defined
(`src/env.ts:32-33`). The lazy `getStore()` retry rebuilds the store/embeddings
from that **already-parsed `cfg`** — it does not re-read `process.env`
(`src/mcp-server.ts:44-57`). The "config fix on disk … visible to the next tool
call" comment there refers to not caching a *rejected* store promise (e.g. a
fixed store file gets re-opened), **not** to re-reading env vars.

Therefore: changing `.env` / `ORACLE_*` requires reconnecting/restarting the MCP
server. Every CLI subcommand calls `loadConfig()` fresh per invocation
(`src/index.ts:51, 65, 114, 145, 168, 205`), so the CLI always reflects current
env — another reason to verify config-sensitive behavior via the CLI.

Verify: reconnect, then call `oracle_list_repos` / `oracle_search`; a
provider/model mismatch against the store fails loud with an
`IndexFingerprintError` naming the expected values
(`src/store/sqlite-store.ts:382-395`, `docs/architecture.md:52-56`).

## The repeated operator trap

Verifying an **ingest / indexing change** through a **running MCP server** mixes
both axes: your `src/` edit to the ingest pipeline is invisible until build +
reconnect (axis 2), while any pre-existing store data is served fresh (axis 1) —
so the server can show old *behavior* for a new reason and you chase a phantom.
Verify ingest changes via the **CLI** instead: `npm run index` runs `tsx
src/index.ts index` (`package.json:22`, `src/index.ts:47-52`), transpiling your
edited source on the spot and driving the same `runIndex` the MCP tool uses
(`src/ingest/runner.ts:3-4`). That isolates "did my ingest change work" from
"is the server running new code." Once the CLI proves the store is correct, the
running MCP server sees the result on its next query with no restart (axis 1).
