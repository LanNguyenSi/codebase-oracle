# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

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
