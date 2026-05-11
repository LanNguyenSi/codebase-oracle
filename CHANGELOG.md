# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- Default skip-dir list now prunes vendored package caches and IDE
  workspaces (`.bun`, `.cache`, `.husky`, `.idea`, `.opencode-home`,
  `.pnpm-store`, `.vscode`, `.yarn`) on top of the prior `node_modules`,
  `dist`, `.next`, etc. Repos that vendored a Bun install cache under
  `.opencode-home/.bun/install/cache` were polluting cross-repo
  auth/token queries with thousands of cached third-party chunks; those
  no longer reach the embedder.
- Default `ORACLE_LLM_MODEL` for the Anthropic provider bumped from
  `claude-sonnet-4-20250514` to `claude-sonnet-4-6`. The previous id
  pointed at the May-2025 Sonnet 4 release which has been superseded by
  the Sonnet 4.x line; new installs were hitting opaque LLM failures
  against a retired model.

### Fixed

- `oracle_query` now reports the SDK error reason instead of swallowing
  it behind `status 500`. The wrapper appends the message body (capped
  to the first non-empty line, 240 chars) and any network code
  (`ECONNREFUSED`, `ENOTFOUND`, etc.) to the formatted detail string,
  so callers see `status 401, request id req_abc, 401 unauthorized`
  rather than a bare status that hides whether the failure was auth,
  model retirement, or context overflow.

### Added

- `ORACLE_SKIP_DIRS`: comma-separated directory names appended to the
  default skip list. Append-only by design, so callers can't
  accidentally unset `node_modules` or `.git` by overriding this value.
- New LLM provider `openai-compatible` plus `ORACLE_LLM_BASE_URL` and
  `ORACLE_LLM_API_KEY`. One lane covers Groq, OpenRouter, Together,
  vLLM, Ollama, and anything else that speaks the OpenAI chat shape.
  Keeps the LLM key isolated from the embedding key so two providers
  can coexist without leaking credentials across lanes.

### Deprecated

- `ORACLE_LLM_PROVIDER=ollama` plus `ORACLE_OLLAMA_BASE_URL` /
  `OLLAMA_API_KEY` still resolve, but the first call against a legacy
  config now logs a one-shot warning. Migrate to the
  `openai-compatible` provider + `ORACLE_LLM_*` env pair; the legacy
  vars will be removed in a future release.
- The fallback in the LLM client that silently picked up
  `OPENAI_API_KEY` when no Ollama key was set is gone. Embedding and
  LLM keys are independent now; a missing LLM key surfaces an
  endpoint-level 401 via the improved error wrapper instead of being
  papered over.

## [0.4.1] - 2026-05-02

Patch release. Two table-hygiene leaks in the full-reindex path that
caused `repo_meta` to drift away from `docs` over time, plus the
infrastructure to keep that gap from regressing. Surfaces as no
functional regression for queries, but `oracle_list_repos` was missing
an `indexed` suffix for any repo last touched in a watcher-less
workflow, and the `repo_meta` table grew unbounded with every deleted
clone.

### Fixed

#### Index hygiene: orphan `repo_meta` rows + reused-only freshness

- `deleteByFile` now drops the `repo_meta` row when the last file of
  a repo is pruned, atomically in the same transaction. Previously it
  always touched the freshness stamp regardless of whether anything
  was left, leaving an orphan row whenever a clone disappeared from
  disk.
- The full reindex (`oracle index`) now stamps every repo that
  yielded ≥1 file in this scan via the new `store.touchRepo(repo, ts)`
  API — even if every file was hash-reused. Watcher-less workflows
  were the only path that wrote to `repo_meta` through a successful
  upsert, so reused-only repos used to render with `last_indexed_at = null`
  forever.
- The reindex command also calls `store.pruneOrphanRepoMeta()` once
  at startup as a backfill sweep for stores that predate the
  `deleteByFile` fix. Idempotent; logs `Cleared N orphan repo_meta
  row(s)` when it actually does work.
- ([#22](https://github.com/LanNguyenSi/codebase-oracle/pull/22))

### Added

#### Stub embedding provider for integration tests

- New `ORACLE_EMBEDDING_PROVIDER=stub` returns deterministic 8-dim
  vectors from `sha256(text)`. Documented test-only and refused under
  `NODE_ENV=production`. Lets the new CLI integration test exercise
  the full reindex pipeline without an OpenAI key.
- New CLI-spawn integration test in `tests/integration/index-cli.test.ts`
  drives `tsx src/index.ts index --path <tmp>` against a fixture
  scanRoot through two scenarios: a reindex with vanished + partial +
  fully-reused repos, and a legacy-store cleanup via the startup sweep.
- ([#23](https://github.com/LanNguyenSi/codebase-oracle/pull/23))

#### Documentation restructure

- README rewritten to a 60-second hook with the rest of the prose
  moved into `docs/`. Faster path to "should I install this".
- ([#21](https://github.com/LanNguyenSi/codebase-oracle/pull/21))

### Security

- `uuid` overridden to `^14.0.0` to pull in the GHSA bounds-check fix
  that surfaced through Dependabot. No direct dependency, but pulled
  in transitively by `@langchain/*`.
- ([#20](https://github.com/LanNguyenSi/codebase-oracle/pull/20))

### Migration

No action required. On the next `oracle index` run, the startup sweep
cleans any legacy orphan `repo_meta` rows; you may see a one-time
`Cleared N orphan repo_meta row(s)` log line — that is expected and
not an error. The MCP server caches `storePromise` after its first
tool call, so reconnect after upgrading and rebuilding `dist/`.

## [0.4.0] - 2026-04-27

Agent-UX release. Three improvements pulled from a dogfood feedback
session that asked "what would make Claude actually use codebase-oracle
as the default for code lookup, instead of falling back to grep?". The
three answers — line numbers in results, freshness signal per repo,
and a way to expand context without leaving the oracle — all ship in
this release.

### Added

#### Line numbers in chunk results

- The splitter now records `lineStart` and `lineEnd` (1-indexed,
  inclusive) on every chunk's metadata. All five rendering sites
  (CLI `search` / `query`, MCP stdio, MCP HTTP, RAG context block,
  raw-context fallback) emit the chunk location as
  `path:lineStart-lineEnd` (or `path:line` for single-line chunks).
- New `formatChunkLocation(metadata)` helper centralises the
  rendering and gracefully falls back to bare `filePath` for chunks
  indexed before this release. Older chunks pick up line numbers on
  their next re-embed; no store migration required.
- ([#16](https://github.com/LanNguyenSi/codebase-oracle/pull/16))

#### Per-repo `last_indexed_at` on `oracle_list_repos`

- New `repo_meta(repo, last_indexed_at)` SQLite table tracks when each
  repo was last touched by `upsertFile` / `insertBatch` /
  `deleteByFile`; `deleteByRepo` drops the row entirely so unindexed
  repos don't surface stale timestamps. All bumps live inside the
  existing `db.transaction` so rollback cleans up.
- `oracle_list_repos` output gains an `(indexed <ISO>, <relative>)`
  suffix per repo where present. Repos last touched before this
  release render bare until their next re-embed.
- New `formatRepoLine` / `formatRelativeFreshness` helpers in
  `src/format-freshness.ts` own the rendering. Optional `prefix`
  parameter so the CLI's two-space indent and the MCP's `- ` bullet
  can share the same renderer.
- ([#17](https://github.com/LanNguyenSi/codebase-oracle/pull/17))

#### `oracle_expand` MCP tool

- New tool reads a window of lines around a position in an indexed
  file. Closes the **search → expand → edit** loop without leaving
  the oracle: paste a `path:line` from `oracle_search` into
  `oracle_expand` and get the surrounding 30 lines back in cat-n
  format, ready to feed into another tool.
- Window default 30 lines, capped at 200. Centered around the
  requested line; clamps cleanly at start and end of file. Trailing
  `\r` stripped so CRLF-encoded files render clean.
- Four typed failure modes (`not_indexed`, `no_absolute_path`,
  `file_missing`, `read_error`) with human-readable messages.
- New `expand <repo> <path>` CLI command with `--line` and `--window`
  options.
- New `getFileMetadata(repo, filePath)` accessor on the SQLite store
  + `VectorStoreWrapper` so the tool can recover the indexed
  `absolutePath` without making the caller know it.
- Honest disclosure in the tool description: reads from disk via the
  indexed `absolutePath`, not from a stored snapshot. If the working
  copy has changed since indexing, the lines may not match what
  `oracle_search` returned. `oracle_list_repos` shows the indexed
  timestamp.
- ([#18](https://github.com/LanNguyenSi/codebase-oracle/pull/18))

### Changed

- `oracle_search` and `oracle_list_repos` output formats now include
  the per-chunk line range and the per-repo indexed timestamp
  respectively. Both changes are additive — clients that ignored
  the suffixes still parse the leading `repo` / `path` cleanly.

## [0.3.0] - 2026-04-17

**Breaking:** on-disk format changed from `embeddings.jsonl` to a SQLite file
(`store.db`) backed by [sqlite-vec](https://github.com/asg017/sqlite-vec).
Either run `npm run migrate-store` to convert in place, or delete
`~/.codebase-oracle/embeddings.jsonl` and re-index.

### Added

#### sqlite-vec vector store
- New on-disk format at `~/.codebase-oracle/store.db`: SQLite tables `meta`,
  `docs`, plus a vec0 virtual table `vectors` for cosine KNN search.
- WAL mode enables concurrent reader + writer, so a running MCP server sees
  watch-mode updates on its next query without a restart. The v0.2.0 known
  limitation is gone.
- Embedding fingerprint now lives in the `meta` table instead of a JSONL
  header line; the load-time compatibility check has the same semantics.
- `oracle_search` / `oracle_query` / `oracle_list_repos` are unchanged from
  the agent's perspective.

#### Incremental watch writes
- `watch` mode now upserts per file directly against the store
  (`DELETE FROM docs/vectors WHERE repo=? AND file_path=?; INSERT ...`) in
  one transaction instead of rewriting the full index on every flush. No
  more multi-second disk churn on heavy edit sessions.

#### Migration
- `npm run migrate-store` converts a v0.2.0 `embeddings.jsonl` into the new
  format, preserves the embedding fingerprint, and moves the JSONL to
  `.embeddings.jsonl.bak` on success. Refuses to run if a non-empty
  `store.db` already exists.

#### Tests
- Unit tests for the raw SQLite store (CRUD, similarity, WAL concurrent
  reader/writer incl. cross-process via `spawnSync`, `initializeSchema`
  contention) and the migration command. Suite grew from 92 to 105 tests.

### Changed
- `createVectorStore`, `listIndexedRepos`, and every ingest/watch path now
  talk to a `SqliteStore` handle. Cold-start RSS for the CLI drops from
  ~1.5 GB (full in-memory index) to ~200 MB (measured on a 49,869-chunk /
  dim-1536 corpus, including Node + tsx wrapper overhead). The store stays
  on disk and similarity search runs against the vec0 table.
- `similaritySearch` is now a SQL KNN query, not a JS linear scan.
- `watch` log lines no longer duplicate the repo segment (`fake-repo/fake-repo/a.ts`
  → `fake-repo/a.ts`).

### Removed
- `embeddings.jsonl` / `embeddings.json` load, persist, append, and
  initialize helpers. The JSONL code path is gone; use `migrate-store` or
  re-index.

## [0.2.0] - 2026-04-17

Adds a safety net around model swaps, a watch mode that keeps the
index fresh without manual re-runs, and opt-in auth for the HTTP
MCP server so it can safely run outside of loopback. Also the first
serious slug of unit tests on the retrieval path.

### Added

#### Embedding fingerprint + load-time guard
- `embeddings.jsonl` meta line now carries `embeddingProvider`,
  `embeddingModel`, and `dimension`.
- On load, the index is checked against the active config and refuses
  to run on a mismatch with a clear instruction to delete the data
  dir or revert the env change. Closes the silent-corruption hole
  where swapping `ORACLE_EMBEDDING_MODEL` produced garbage scores or
  `NaN` cosine values without warning.
- Defense-in-depth dimension check in `similaritySearch` for legacy
  indexes that pre-date the fingerprint.
- `addDocuments` refuses to mix dimensions mid-session.
- Legacy indexes load with a warning pointing to `npm run index` as
  the upgrade path.
- HTTP MCP server propagates `IndexFingerprintError.message` instead
  of swallowing it as `"Internal error"`.
- CLI exits cleanly (no stacktrace) on a fingerprint mismatch.

#### Watch mode (`npm run watch`)
- Debounced (default 3 s) chokidar watcher on the scan root.
- File add/change → re-embed only the touched file; delete → drop
  vectors; repo root removal → purge all of its vectors; a new
  top-level directory containing `.git` is registered as a new repo
  (back-fill it once with `npm run index`, subsequent edits flow
  through watch).
- Save-storms collapse into a single re-embed thanks to chokidar's
  `awaitWriteFinish` + the debounce dedup.
- Embed is atomic: new vectors are computed first, old vectors are
  only swapped in on success.
- Known limitation: running stdio / HTTP MCP servers do not hot-reload
  the store — restart to pick up changes.

#### HTTP MCP auth (opt-in)
- `ORACLE_HTTP_TOKEN` — when set, every `POST /mcp` request must carry
  `Authorization: Bearer <token>`; compared in constant time. `GET
  /health` stays open.
- `ORACLE_HTTP_BIND` — override the bind address (default
  `127.0.0.1`). Any value outside `{127.0.0.1, localhost, ::1}`
  requires `ORACLE_HTTP_TOKEN` or the server refuses to start, so
  there is no accidental off-loopback exposure.

#### Tests
- 21 unit tests for `src/retrieval/chain.ts` helpers (`createLlm`,
  `getLlmErrorDetails`, `extractSources`, `formatRawContextAnswer`).
  Covers provider selection, error formatting, source dedup, and
  raw-context rendering — no real API calls.
- Fingerprint, HTTP auth, and watch-mode tests added. Suite grew from
  26 to 92 tests.

### Changed
- `createVectorStore` now runs `assertCompatibleIndex` at
  construction and exposes the fingerprint error to every caller
  (CLI, stdio MCP, HTTP MCP).

### Removed
- `CLAUDE.md` — redundant with `README.md`; Claude Code falls back to
  the README when `CLAUDE.md` is absent.

## [0.1.0] - 2026-04-16

Initial release. codebase-oracle is a shared semantic index over local
git repos, designed agent-first: one scan, many Claude Code / MCP
sessions reuse the same vector store instead of scanning and embedding
on their own.

### Added

#### Ingest + retrieval core
- Scanner that walks every git repo under a root directory, filters by
  file type, skips `node_modules` / `dist` / `build` / `.git` / large
  files, and streams `ScannedFile` records with content hashes.
- Language-aware splitter that chunks source files on function/class
  boundaries.
- In-memory vector store with on-disk JSONL persistence under
  `~/.codebase-oracle/`, cosine similarity search, and metadata
  filtering.
- Incremental indexing: unchanged files are reused from persisted
  vectors via file-hash match; only new or changed files are
  re-embedded. Batch-by-batch checkpoints so interrupted runs resume
  without redoing completed work.

#### Providers
- OpenAI embeddings (`text-embedding-3-small` by default) and
  LLMs (`gpt-4o-mini` fallback under `auto`).
- Anthropic Claude for answer generation (preferred under `auto`).
- Ollama as a drop-in OpenAI-compatible provider for both embeddings
  and LLM — enables fully local operation with `nomic-embed-text` +
  `llama3.1`.

#### CLI (`src/index.ts`)
- `index` — full or incremental scan + embed + persist.
- `query` — retrieval-augmented question answering with source
  citations, optional `--repo` and `--limit` filters.
- `search` — raw vector similarity search without LLM interpretation.

#### MCP server (stdio, `src/mcp-server.ts`)
- Three tools exposed to Claude Code: `oracle_query`, `oracle_search`,
  `oracle_list_repos`.
- Lazy store initialization so registration is cheap and the first
  tool call triggers index load.
- `oracle_list_repos` reports repos actually present in the vector
  index with chunk and file counts (backed by
  `VectorStoreWrapper.listRepos()`), not just directories on disk.

#### HTTP MCP server (`src/http-server.ts`)
- Streamable HTTP MCP transport, bound to `127.0.0.1:3100` by default
  (override with `ORACLE_HTTP_PORT`).
- Same three tools as the stdio server, shared singleton.
- Health endpoint at `GET /health`.

#### Scanner defaults
- Default file-extension allowlist covers JS/TS plus sibling languages
  (`.py`, `.php`, `.go`, `.rs`, `.java`, `.vue`) and config/infra
  (`.yaml`, `.yml`, `.toml`, `.sql`, `.prisma`, `.sh`). The built-in
  manifest filter still keeps random `.json` files out by default
  but only applies when defaults are in use.
- `ORACLE_INCLUDE_EXTENSIONS` env var overrides the allowlist entirely
  (comma-separated, leading dot optional). When set, a manifest `.json`
  is no longer filtered.

#### Docs
- Agent-first README: the MCP use case is framed as primary, the CLI
  as secondary. Example agent prompts, clear split between agent and
  human flows.
- Credits to [andrepester/rag-search-mcp](https://github.com/andrepester/rag-search-mcp)
  as the conceptual inspiration.

#### Release infrastructure
- This release introduces `.github/workflows/release.yml`, triggered
  on `v*` tags. It reuses `ci.yml` via `workflow_call`, extracts this
  CHANGELOG section for the tag, and publishes the GitHub Release via
  `softprops/action-gh-release@v2`.
