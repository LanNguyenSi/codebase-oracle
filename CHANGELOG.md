# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.10.2] - 2026-07-13

### Fixed

- **Sources-expansion no longer displaces a below-parent organic hit off the result list entirely.** The dedup rule used to be organic-anywhere-wins-by-skipping: if a parent's pointed-at file already appeared as an organic hit anywhere in the candidate list, its injection was skipped outright, leaving the file's fate to whichever rank it happened to occupy. A later, non-ground-truth sibling injection could (and, per the OKF benchmark's Q12 regression, did) fill the freed slot and push the organic row past the caller's `limit` cut before the outer loop ever reached it. The rule is now hoist-not-skip: a pointed-at file that is an organic hit ranked at or above its pointing parent stays exactly where it is, no duplicate, no reorder. A pointed-at file ranked below its pointing parent (not yet placed, whether or not it would have survived the `limit` cut on its own) is hoisted into the injection slot right after the parent, preserving its organic content/snippet (no synthesized `[expanded from ...]` stub). Fixes codebase-oracle `d165ff85`; live-verified against the exact Q12 query (agent-tasks index): `backend/src/routes/tasks.ts` moved from rank 6 (past the display cut) to rank 2, carrying its real matched chunk instead of a generic first-chunk stub.
- **Test files that drift out of type sync with `src/` no longer go unnoticed.** Six `tests/unit/*.test.ts` local `Config` test helpers (chain, embeddings, migrate-store, query-codebase, sqlite-store, vector-store) predated `maxFileSizeBytes` becoming a required `Config` field (0.10.0) and failed `tsc --noEmit`, but nothing ran that check: `npm run build` only typechecks `src/`, and vitest runs tests through esbuild without typechecking. All six helpers now set `maxFileSizeBytes`, and CI gains a "Type check tests" step (`npx tsc --noEmit -p tsconfig.tests.json`, also runnable locally as `npm run test:typecheck`) so a future required-field addition fails the build instead of silently rotting the test suite's types. The new `tsconfig.tests.json` excludes `tests/eval/corpus/`, which is vendored fixture text walked by the eval runner, not real project source.

## [0.10.1] - 2026-07-05

### Fixed

- **A repo-drop as the very first watch event no longer crashes with `IndexFingerprintError`.** `deleteByRepo` now applies the same never-initialized-store guard as its sibling `deleteByFile`: on a fresh watch whose first `flush()` drains a dropped repo before any file was embedded, the vec delete is skipped (the store has no vec0 table yet) instead of routing through the unprepared vec statements.

## [0.10.0] - 2026-07-04

### Added

- **`ORACLE_MAX_FILE_SIZE` config option** (bytes, default `500000`) makes the per-file size ceiling explicit and overridable, instead of a hardcoded constant.
- **Loud per-file skip reporting.** `npm run index` now reports every file it skips for size or fails to read, one `WARNING:` line per file naming the path and the reason, plus a one-line total, on stderr. `npm run watch` reports the same way through its console logging when a changed file trips the limit.
- **`IndexSummary` gains `filesSkipped` and `skippedFiles`**, and `formatIndexSummary` (surfaced by the MCP `oracle_reindex` tool) now includes the skip count in its files segment.

### Changed

- **The per-file size limit moved from an implicit `content.length > 200_000` (UTF-16 chars, read into memory first) to a `stat`-first check in true bytes against the configurable `ORACLE_MAX_FILE_SIZE` (default 500 KB)**, in both the scanner and `watch.ts`, which previously duplicated the old limit as its own `MAX_FILE_BYTES` constant and now reads it from config.

### Fixed

- **Files over the size limit no longer disappear from the index without a trace.** `agent-tasks/backend/src/routes/tasks.ts` (207,716 bytes) was silently dropped by the old `content.length > 200_000` check; it and any file like it are now reported, not swallowed.
- **A per-file read error (permission denied, binary decode failure) no longer vanishes into a bare `catch {}`.** It is now reported the same way as a size skip.

## [0.9.0] - 2026-07-04

### Added

- **`oracle_search` now expands OKF `sources:` pointers at search time (default on).** When a retrieved chunk carries `fmSources` (the repo-root-relative paths from its markdown frontmatter's `sources:` field), those files are treated as vouched-for: a representative chunk (the pointed-at file's first chunk, same repo) is injected into the result list immediately after its parent row, in `fmSources` order, at most 3 per parent. Injected rows render an additive `[expanded from <parent basename>]` header marker (alongside any `[type]` tag) so provenance is legible. Dedup is per `(repo, filePath)`: a file already present as an organic hit anywhere, or already injected, is skipped — organic always wins. The list stays capped at the caller's `limit`, so injections displace tail organic rows. Non-resolving pointers (a directory, a glob, a typo, or an unindexed file) are skipped silently and deterministically. Available on the CLI, stdio MCP, and HTTP MCP. For a corpus where no retrieved chunk carries `fmSources` (or when expansion is disabled), the result list and its rendered output are byte-identical to v0.8.0 — the `expandedFrom` marker is transient, living only on the returned Document, never persisted.
- **Opt-out: `--no-expand-sources` on the CLI `search` command and `expand_sources: false` on the `oracle_search` MCP/HTTP tool** disable injection entirely and return the raw retrieval result. This is the M2 motivation — one search now surfaces both the doc that describes a subsystem and the implementation files it points at, without a second round-trip.
- **New `getFirstChunkByFile(repo, filePath)` store accessor** on the SQLite store and `VectorStoreWrapper` returning the first chunk of a file (lowest rowid, i.e. top of file) as `{ pageContent, metadata }`, backing the expansion.

## [0.8.0] - 2026-07-03

### Added

- **`oracle_search` gains `type` and `tags` filters over OKF frontmatter chunk metadata** (`fmType` / `fmTags`, added in v0.7.0). `type` is a strict-equality match; `tags` requires ALL listed tags to be present. Both only match chunks that HAVE the corresponding field, so chunks without frontmatter metadata are excluded whenever a filter is set, and both AND-compose with the existing `repo` / `path_glob` filters. Available on the CLI (`-t, --type`, `--tags` comma-separated), stdio MCP, and HTTP MCP.
- **Search results show OKF metadata when present.** A matching chunk's header gains a `[type]` tag (e.g. `[3] docs/okf/backend.md:1-40 (agent-tasks) [module]`) and, when `fmSources` is set, an additional `sources: path1, path2` line. Chunks without frontmatter metadata render byte-identical to before.
- **`oracle_query` answers get an automatic `Pointers (from OKF sources metadata):` section**, mechanically assembled (no LLM) from the deduped, rank-ordered union of `fmSources` across every retrieved chunk, appended after the existing sources list and capped at 10 entries with a truncation note. Omitted entirely when no retrieved chunk carries `fmSources`. No new params needed on `oracle_query`.

## [0.7.0] - 2026-07-03

### Added

- **Markdown YAML frontmatter is now extracted into chunk metadata at ingest.** Files whose content leads with a `---`/`---` block get `fmType`, `fmTitle`, `fmTags`, and `fmSources` metadata keys (read from the frontmatter's `type`, `title`, `tags`, and `sources` fields), each included only when present with the right type; anything else, or a malformed block, is fail-soft (a single logged warning, no `fm*` keys, ingest continues as plain text). The chunked content itself is unchanged, only metadata is added. This is groundwork for OKF-aware retrieval; the retrieval and MCP/HTTP layers do not read these keys yet. No on-disk migration: existing chunks lack the new keys until their file changes or the store is rebuilt (see [docs/upgrades.md](docs/upgrades.md)).

## [0.6.5] - 2026-06-16

Patch release. Security bump for esbuild advisories via tsx, an expanded eval set for retrieval quality, and internal housekeeping.

### Security

- **Bump tsx to ^4.22.4** clearing two esbuild advisories: GHSA-gv7w-rqvm-qjhr and GHSA-g7r4-m6w7-qqqr (PR #51).

### Changed

- **Version is now read from package.json** as the single source of truth; the hardcoded literal that had drifted out of sync is gone (PR #50).
- **Eval set expanded from 4 to 20 hand-labelled Q&A pairs** across four new toy-corpus fixtures (`db-toy`, `form-toy`, `queue-toy`, `server-toy`). The baseline is updated to match; regressions still fail the run (PR #53).
- **README now surfaces the manual pre-release eval gate**: the required `npm run eval` check before cutting a release is documented with a description of what a passing run means (PR #52).

## [0.6.4] - 2026-06-09

Security release closing a HIGH audit finding and a CVE sweep.

### Security

- **HIGH: the HTTP MCP server now builds a fresh server per request** (PR #47). The `POST /mcp` handler shared one singleton `McpServer` across all requests: the first request set `server._transport` via `connect()`, a stateless POST never clears it (the SDK only resets it in `_onclose`, which a normal POST does not trigger), so every subsequent request threw "Already connected to a transport" and was returned as a generic `-32603` error. The server served exactly one request before failing for all clients. A `buildServer()` factory now constructs a new `McpServer` (plus transport) per request, matching the SDK's stateless pattern; tools still close over the shared lazy `getStore()` / config so there is no per-request store rebuild, and the per-request server and transport are closed once the response is fully streamed.
- **hono advanced to `4.12.23`** (4 MEDIUM CVEs: CVE-2026-47673 / 47674 / 47675 / 47676, PR #48). Patched in hono 4.12.21.

## [0.6.3] - 2026-05-28

### Fixed

- CLI commands `query`, `search`, `list-repos`, and `mcp` no longer
  crash with a raw `ZodError: scanRoot is required` when
  `ORACLE_SCAN_ROOT` is not set in the environment. Commit `4827b3d`
  (v0.6.0 release prep) tightened the Zod schema so `loadConfig()`
  refused to start without `ORACLE_SCAN_ROOT`, but only `index` and
  `watch` actually consume `config.scanRoot`. Read-only commands now
  load config without requiring the env var; `index` and `watch` get
  the same friendly "ORACLE_SCAN_ROOT is required, set it to the
  directory containing your git repos" message via a new
  `assertScanRoot()` helper that runs at the top of `runIndex` and
  `runWatchMode`. PR #45, agent-tasks `d8df9a5f`.
- CLI `--version` and the MCP server handshake now both report
  `0.6.3` instead of the stale `0.6.1` literal that had drifted out
  of sync with `package.json` since v0.6.0.

## [0.6.2] - 2026-05-24

### Fixed

- `getLlmErrorDetails` now walks `err.errors[]` when neither `err.code`
  nor `err.cause?.code` is present, surfacing the network code for
  undici dual-stack `AggregateError` wrappers (e.g. `ECONNREFUSED` on
  both IPv4 + IPv6 connect attempts). Previously these wrappers landed
  in the agent-facing error as a bare `fetch failed` with no code,
  which made it hard to distinguish a misconfigured base URL from a
  briefly-down service. Aggregate child `.message` is also folded into
  the message picker as a fallback, and the `??` chain on `message`
  is now `||` so an empty top-level `message: ""` falls through to
  cause / aggregate children. Verified with three new unit tests in
  `tests/unit/chain.test.ts` (happy path, top-level code beats
  aggregate, child without code is skipped); the existing
  `{ message: "" }` → `null` test confirms the chain change doesn't
  regress the no-fallback path.

  PR #40, agent-tasks `a97d35a3`.

## [0.6.1] - 2026-05-16

Hotfix for 0.6.0. The tag-driven publish workflow for 0.6.0 was rejected
by npm with `Error verifying sigstore provenance bundle: Failed to
validate repository information: package.json: "repository.url" is
"git@github2:LanNguyenSi/codebase-oracle.git", expected to match
"https://github.com/LanNguyenSi/codebase-oracle" from provenance`. The
local SSH host alias `github2` could not be matched against the
GitHub-Actions provenance attestation, which records the canonical
`https://github.com/LanNguyenSi/codebase-oracle` repo URL. Switching
`repository.url` to the canonical `git+https://github.com/...` form
satisfies the provenance check.

### Changed

- `package.json` `repository.url` now uses the canonical
  `git+https://github.com/LanNguyenSi/codebase-oracle.git` form
  instead of the local SSH host alias. Required for npm
  publish-with-provenance to validate. Local `git push` keeps working
  because Git remote URLs are configured per-clone, independent of
  `package.json`.

## [0.6.0] - 2026-05-16

Packaging release: the project is now published to npm as
`@lannguyensi/codebase-oracle`. The unscoped name on npm belongs to an
unrelated CLI, which made `npm i -g codebase-oracle` resolve to the wrong
package and blocked the harness Full template from listing this MCP
server as a default dependency. The scoped name fixes that, and the CLI
gains an explicit `mcp` subcommand so a manifest entry like
`command: [codebase-oracle, mcp]` boots the MCP server end-to-end.

### Added

- `codebase-oracle mcp` subcommand on the CLI. Starts the MCP server
  over stdio by delegating to `src/mcp-server.ts`. Equivalent to the
  existing `npm run mcp` developer flow, but reachable from a
  globally-installed binary so harness manifests and other MCP clients
  can wire it in without a local source checkout.
- `.github/workflows/release.yml` now publishes the npm package on `v*`
  tag push, with npm provenance and a guard that fails the job if the
  tag does not match `package.json` `version`. Requires the `NPM_TOKEN`
  repo secret.
- `package.json` `files` allowlist (`dist`, `README.md`, `LICENSE`,
  `CHANGELOG.md`) and `prepublishOnly: npm run build` so the published
  tarball ships a clean, freshly-built `dist/` and nothing else.

### Changed

- `package.json` `name` is now `@lannguyensi/codebase-oracle`; the bin
  name stays `codebase-oracle`. The previously-published unscoped
  `codebase-oracle` package on npm is an unrelated CLI whose bin is
  named `oracle`, so the two can coexist when installed globally.
- `src/mcp-server.ts` now exports `startMcpServer()`; the legacy
  self-start path (`node dist/mcp-server.js`, `npm run mcp`) keeps
  working via an `import.meta.url === process.argv[1]` guard.

## [0.5.0] - 2026-05-11

Minor release. Substantial new agent-facing surface: a generic
`openai-compatible` LLM provider lane (Groq, OpenRouter, Together,
vLLM, local Ollama, all through one set of env vars), an
`oracle_reindex` MCP verb so agents can make freshly-merged code
visible without waiting on the scheduled background reindex, a
`path_glob` filter on `oracle_search` to express structural
cross-repo queries like "every release.yml workflow", and a vendored
eval framework that catches retrieval-quality regressions per
release. Plus the boring cleanup that makes the rest land cleanly:
the LLM error wrapper now surfaces the real SDK reason instead of
opaque `status 500`, the scanner prunes vendored package caches by
default, and the legacy ollama-named env vars are kept as a
deprecated alias.

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
- New MCP tool `oracle_reindex`. Runs the same incremental pipeline as
  `npm run index` from inside an agent session, closes the in-process
  store handle first so the indexer doesn't fight itself for the
  SQLite write lock, and returns a one-line summary of files scanned
  / changed / pruned and chunks reused / embedded.
- Systemd user-timer templates under `scripts/systemd/`. Drop into
  `~/.config/systemd/user/`, edit the `WorkingDirectory` to point at
  your checkout, and `systemctl --user enable --now
  codebase-oracle-index.timer` for a daily background reindex. Default
  schedule is `04:00` local with a 15-minute random delay and
  `Persistent=true` so a missed window (asleep laptop) catches up on
  next boot.
- Retrieval-quality eval framework under `tests/eval/`: a vendored
  3-repo fixture corpus (`auth-toy`, `cli-toy`, `config-toy`), four
  hand-labelled Q&A pairs in `questions.json`, a `npm run eval`
  runner that indexes the corpus, runs each question through
  `oracle_search`, compares the per-question pass set against
  `baseline.json`, and exits non-zero on regressions. Use
  `npm run eval -- --update` to bake in an intentional improvement.
- `oracle_search` accepts an optional `path_glob` argument. picomatch
  semantics: `*` within a segment, `**` recursive, `?` single char,
  `{a,b}` alternatives. AND-composes with the existing `repo` filter.
  Closes the loop on structural cross-repo queries: `query="release"`,
  `path_glob="**/.github/workflows/*.yml"` returns only workflow files.
  Also wired into the CLI as `--path-glob`.
- Scanner now respects a `.codebase-oracle-skip` sentinel file:
  any directory containing one is pruned wholesale, regardless of
  name. Lets vendored fixtures (`tests/eval/corpus/`) and other
  "lives in the source tree but should never enter the index"
  subtrees stay co-located with the code that owns them. Documented
  in `docs/configuration.md`.

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
