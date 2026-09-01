# Configuration

All configuration is via environment variables. The CLI and MCP server auto-load `.env` from the current working directory if present (the repo root when run via `npm run` scripts).

## Prerequisites

- Node.js 22+
- `OPENAI_API_KEY` (required if `ORACLE_EMBEDDING_PROVIDER=openai`)
- `ANTHROPIC_API_KEY` (optional, for Claude-powered answers)
- A running Ollama instance locally if you want `ORACLE_*_PROVIDER=ollama`

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ORACLE_EMBEDDING_PROVIDER` | No | `openai` | Embedding provider: `openai` or `ollama`. `stub` exists for tests only (deterministic 8-dim vectors, no network) and is refused under `NODE_ENV=production` |
| `ORACLE_LLM_PROVIDER` | No | `auto` | LLM provider: `auto`, `anthropic`, `openai`, `openai-compatible`, `ollama`. `openai-compatible` is the generic lane for Groq, OpenRouter, Together, vLLM, etc.; `ollama` is kept as a deprecated alias for the same lane (prints a one-shot warning) |
| `ORACLE_LLM_BASE_URL` | Conditionally | — | Required when `ORACLE_LLM_PROVIDER=openai-compatible`. The provider's OpenAI-compatible `/v1` endpoint (e.g. `https://api.groq.com/openai/v1`) |
| `ORACLE_LLM_API_KEY` | No | — | API key for the `openai-compatible` lane. Leave unset for endpoints that don't require auth (local Ollama, some vLLM setups); the SDK surfaces a 401 if one is required. Kept separate from `OPENAI_API_KEY` so embedding and LLM keys never cross-leak |
| `OPENAI_API_KEY` | Conditionally | — | Required when `ORACLE_EMBEDDING_PROVIDER=openai`; also used for OpenAI LLM |
| `OPENAI_BASE_URL` | No | — | Override OpenAI-compatible base URL for OpenAI provider |
| `ANTHROPIC_API_KEY` | No | — | Anthropic API key for answer generation |
| `OLLAMA_API_KEY` | No | — | Optional API key for Ollama provider (defaults to `ollama`) |
| `ORACLE_OLLAMA_BASE_URL` | No | `http://localhost:11434/v1` | Ollama OpenAI-compatible base URL. The plain `OLLAMA_BASE_URL` env var is honored as a fallback alias when `ORACLE_OLLAMA_BASE_URL` is unset |
| `ORACLE_SCAN_ROOT` | Yes (for `index` / `watch`) | — (none) | Root directory to scan for git repos. There is no default: the writer commands (`index`, `watch`) exit with `ORACLE_SCAN_ROOT is required` when it is unset. Read-only commands (`query`, `search`, `list-repos`) never read it; the stdio `mcp` server reads it only when its `oracle_reindex` tool is invoked (which runs the indexer and so requires it) |
| `ORACLE_DATA_DIR` | No | `~/.codebase-oracle` | Directory for persisted index data |
| `ORACLE_EMBEDDING_MODEL` | No | `text-embedding-3-small` (OpenAI) / `nomic-embed-text` (Ollama) | Embedding model name for selected provider |
| `ORACLE_LLM_MODEL` | No | `claude-sonnet-4-6` (`auto`/Anthropic), `gpt-4o-mini` (OpenAI), `llama3.1` (Ollama) | LLM model name for selected provider |
| `ORACLE_VECTOR_STORE` | No | `directory` | `directory` (persisted) or `memory` (ephemeral) |
| `ORACLE_INCLUDE_EXTENSIONS` | No | _see scanner defaults_ | Comma-separated extension allowlist, replaces defaults entirely (e.g. `.ts,.py,.rb`). Leading dot optional. If you include `.json`, the built-in manifest filter (only `package.json`/`tsconfig.json`) is bypassed: you'll get every matching JSON file. |
| `ORACLE_SKIP_DIRS` | No | — | Comma-separated directory names to skip on top of the built-in defaults (see below). Append-only: defaults like `node_modules` and `.git` are always skipped, so this field is for repo-specific additions (`generated`, `fixtures`, etc). |
| `ORACLE_MAX_FILE_SIZE` | No | `500000` | Per-file size ceiling in bytes, for every extension except the text/doc types listed under `ORACLE_MAX_TEXT_FILE_SIZE`. Files over this size are skipped and reported (never silently dropped — see below). Must be a positive integer; unset (or set to the empty string) falls back to the default, but a set value that isn't a positive integer (e.g. `abc`, `0`, `-5`) fails config loading loudly instead of silently falling back. |
| `ORACLE_MAX_TEXT_FILE_SIZE` | No | `2000000` | Per-file size ceiling in bytes for text/doc file types (currently just `.md`), applied instead of `ORACLE_MAX_FILE_SIZE`. Rationale: the size ceiling exists to bound how much a single file costs to read fully into memory before chunking, and a multi-hundred-KB markdown file (a CHANGELOG, a design doc) is harmless there — the splitter chunks it downstream regardless of file size — so text/doc types get a larger ceiling than the one sized for arbitrary source files. Same positive-integer / fail-loud-on-invalid-value contract as `ORACLE_MAX_FILE_SIZE`. |
| `ORACLE_HTTP_PORT` | No | `3100` | Port for the HTTP MCP server (`npm run serve`) |
| `ORACLE_HTTP_BIND` | No | `127.0.0.1` | Bind address for the HTTP MCP server. Any non-loopback value (e.g. `0.0.0.0`, LAN IP, IPv6 `::`) requires `ORACLE_HTTP_TOKEN`: the server refuses to start otherwise |
| `ORACLE_HTTP_TOKEN` | No | — | Bearer token for the HTTP MCP server. When set, every `POST /mcp` request must carry `Authorization: Bearer <token>` (constant-time compare). `GET /health` stays open |

## Default scan filters

`npm run index` scans all git repos under `ORACLE_SCAN_ROOT`. By default it loads JS/TS sources (`.ts`, `.tsx`, `.js`, `.jsx`, `.vue`), docs (`.md`), sibling languages (`.py`, `.php`, `.go`, `.rs`, `.java`), config/infra (`.yaml`, `.yml`, `.toml`, `.sql`, `.prisma`, `.sh`), and the `package.json` / `tsconfig.json` manifests. Files over `ORACLE_MAX_FILE_SIZE` bytes (default 500 KB) are skipped, except `.md` files, which get the larger `ORACLE_MAX_TEXT_FILE_SIZE` ceiling (default 2 MB) instead. Empty files are silently skipped (nothing to index); files over the applicable size limit and files that fail to read (permission errors, binary decode failures) are each reported on stderr at index time, one line per file naming the path, the reason, and the specific env var whose limit was exceeded, followed by a one-line total — never a silent drop. `npm run watch` reports the same way through its console logging when a changed file trips the limit, and applies the same per-type ceiling. A repo's skipped-file count from the last index run (broken down by size vs. read-error, with a few example paths) is also visible per repo via `oracle_list_repos` / `list-repos` when anything was skipped.

### Default skip directories

The scanner prunes these directory names at any depth so vendored package caches and build output never reach the embedder:

- VCS / language runtimes: `.git`, `__pycache__`, `.venv`
- Build / cache output: `build`, `coverage`, `dist`, `.cache`, `.next`, `.nyc_output`, `.turbo`
- Package managers and their stores: `node_modules`, `.bun`, `.pnpm-store`, `.yarn`, `vendor`
- IDE / workspace caches: `.husky`, `.idea`, `.opencode-home`, `.vscode`

Override the extension allowlist with `ORACLE_INCLUDE_EXTENSIONS`. Add repo-specific directory names to the skip list with `ORACLE_SKIP_DIRS` (appended to the defaults above).

### Per-subtree opt-out: `.codebase-oracle-skip`

For "lives in the source tree but must not enter the index" subtrees (vendored fixtures, generated golden files, sample apps shipped for documentation purposes), drop a `.codebase-oracle-skip` file into the directory. The scanner prunes any subtree that contains this sentinel, regardless of the directory name. The eval-set fixture corpus under `tests/eval/corpus/` uses this so the toy repos never pollute the user's main index.

## OpenAI-compatible providers (Groq, OpenRouter, Together, vLLM, Ollama…)

Any inference endpoint that speaks the OpenAI `chat/completions` shape can serve the LLM step. Set the provider to `openai-compatible` and point the new env vars at the endpoint:

```bash
# Groq (fast hosted llama / kimi / gpt-oss)
export ORACLE_LLM_PROVIDER=openai-compatible
export ORACLE_LLM_BASE_URL=https://api.groq.com/openai/v1
export ORACLE_LLM_API_KEY=gsk_...
export ORACLE_LLM_MODEL=llama-3.3-70b-versatile
```

```bash
# Local Ollama (no key needed)
export ORACLE_LLM_PROVIDER=openai-compatible
export ORACLE_LLM_BASE_URL=http://localhost:11434/v1
export ORACLE_LLM_MODEL=llama3.1
```

```bash
# OpenRouter (aggregates many models behind one endpoint)
export ORACLE_LLM_PROVIDER=openai-compatible
export ORACLE_LLM_BASE_URL=https://openrouter.ai/api/v1
export ORACLE_LLM_API_KEY=sk-or-...
export ORACLE_LLM_MODEL=anthropic/claude-sonnet-4-6
```

The embedding lane stays independent. With `ORACLE_EMBEDDING_PROVIDER=openai` and `OPENAI_API_KEY=sk-…`, embeddings still hit OpenAI; only the LLM step routes through `ORACLE_LLM_*`. Setting both to point at Ollama keeps the entire pipeline local.

## Keeping the index fresh

The store is incremental: only changed and new files re-embed on each run, deleted files prune. Two ways to keep it current without manual `npm run index` calls:

### Scheduled reindex (systemd user timer)

Ship templates live under `scripts/systemd/`. Copy them into `~/.config/systemd/user/`, edit the `WorkingDirectory` path to your checkout, then:

```bash
mkdir -p ~/.config/systemd/user
cp scripts/systemd/codebase-oracle-index.service.example \
   ~/.config/systemd/user/codebase-oracle-index.service
cp scripts/systemd/codebase-oracle-index.timer.example \
   ~/.config/systemd/user/codebase-oracle-index.timer
# edit the WorkingDirectory in the .service file
systemctl --user daemon-reload
systemctl --user enable --now codebase-oracle-index.timer
```

Inspect with `systemctl --user list-timers codebase-oracle-index.timer` and `journalctl --user -u codebase-oracle-index.service --since today`. The default schedule is daily at 04:00 local with a 15-minute random delay; `Persistent=true` catches up after a missed window (laptop asleep).

On a machine that serves as the index source of truth, `ExecStart` may point at `scripts/oracle-refresh.sh` instead of `npm run index` directly (e.g. `ExecStart=/home/CHANGE_ME/git/codebase-oracle/scripts/oracle-refresh.sh`), for the same pull-then-index behaviour described below.

### Scheduled refresh (macOS launchd)

On a machine that serves as the index source of truth for several repos, `scripts/oracle-refresh.sh` fast-forwards every clean checkout directly under `ORACLE_SCAN_ROOT` before running the incremental index. A checkout is skipped (not touched) when it is dirty, unreadable, has a detached `HEAD`, or its current branch has no upstream; a failed `git pull --ff-only` is reported with the first line of git's stderr and the loop continues rather than aborting the run. Hidden directories directly under `ORACLE_SCAN_ROOT` are not seen by the pull loop, matching the indexer's own repo discovery (`discoverRepos` in `src/ingest/scanner.ts` also skips dotfiles). The index step always runs afterward and its exit status becomes the script's exit status.

Two env knobs beyond the usual configuration:

- `ORACLE_REFRESH_PULL=0` skips the pull phase entirely and only runs the index step.
- `ORACLE_REFRESH_INDEX_CMD` overrides the index command (defaults to `npm run index`), run via `bash -c` in the checkout directory. This is operator-controlled and executed as a shell command; treat it as the same trust tier as the plist or unit file that invokes the script.

Ship template lives under `scripts/launchd/`. Copy it into `~/Library/LaunchAgents/`, replace the `CHANGE_ME_CHECKOUT` and `CHANGE_ME_HOME` placeholders with absolute paths, then:

```bash
mkdir -p ~/Library/LaunchAgents
mkdir -p ~/Library/Logs/codebase-oracle
cp scripts/launchd/com.codebase-oracle.refresh.plist.example \
   ~/Library/LaunchAgents/com.codebase-oracle.refresh.plist
# edit the CHANGE_ME_CHECKOUT and CHANGE_ME_HOME placeholders
launchctl bootstrap gui/$UID ~/Library/LaunchAgents/com.codebase-oracle.refresh.plist
```

Inspect with `launchctl print gui/$UID/com.codebase-oracle.refresh` and by tailing `~/Library/Logs/codebase-oracle/refresh.out.log`. Unload with `launchctl bootout gui/$UID/com.codebase-oracle.refresh`. The default interval is hourly (`StartInterval=3600`); `RunAtLoad` also fires one run immediately at login and at bootstrap time, in addition to the interval ticks. `.env` is read from `WorkingDirectory` by the CLI's own dotenv loader, so no secrets go into the plist.

The plist's `PATH` is a plain launchd session `PATH`, not a login-shell one; if `npm` lives under nvm/fnm/volta, edit it in to avoid a "command not found" failure, or point `ORACLE_REFRESH_INDEX_CMD` at the absolute node binary and `dist/index.js`, mirroring the systemd unit's own PATH note above.

### On-demand from an agent (MCP `oracle_reindex`)

The MCP server exposes `oracle_reindex` (no arguments). Agents can call it after merging a relevant PR so the new chunks land in the index without waiting for the next scheduled run. The verb closes the live store handle, runs the same incremental pipeline as `npm run index`, and returns a one-line summary (`Reindex complete in 8.7s. Repos: 39, files: …`). Subsequent `oracle_search` / `oracle_query` calls reopen the store transparently.

### Legacy ollama variables

`ORACLE_LLM_PROVIDER=ollama` plus `ORACLE_OLLAMA_BASE_URL` and `OLLAMA_API_KEY` still work and resolve to the same lane. The first call against a legacy config logs a one-shot deprecation warning. New env names (`ORACLE_LLM_BASE_URL`, `ORACLE_LLM_API_KEY`) take precedence when both are set, so you can migrate without an outage.
