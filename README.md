# codebase-oracle

Semantic search across all your local repos, via MCP or CLI.

[![CI](https://github.com/LanNguyenSi/codebase-oracle/actions/workflows/ci.yml/badge.svg)](https://github.com/LanNguyenSi/codebase-oracle/actions/workflows/ci.yml)

codebase-oracle builds one semantic index over every git repo under a root directory, then exposes it to agents via MCP or to humans via CLI. The vector store lives on your machine; embeddings are computed by OpenAI by default, or fully local via Ollama (configurable). Indexing is incremental: only new and changed files are re-embedded. Built for agents first, humans second.

## How it works

One incremental index over every repo under `ORACLE_SCAN_ROOT`, reachable by agents over MCP and by humans over the CLI, both served from a shared sqlite-vec store.

```mermaid
flowchart LR
    subgraph access["Access"]
        direction TB
        agent["Claude Code / MCP client"]
        human["Human · CLI"]
    end

    agent -->|"oracle_search · oracle_query · oracle_expand · oracle_list_repos<br/>oracle_reindex (stdio only)"| mcp["mcp-server.ts · stdio · all 5 tools<br/>http-server.ts · :3100 · 4 tools"]
    human -->|"index · search · query · expand · list-repos"| cli["index.ts · CLI"]

    subgraph indexing["Indexing pipeline · incremental"]
        direction LR
        scan["ingest/scanner.ts"] --> split["ingest/splitter.ts"] --> embed["store/embeddings.ts<br/>OpenAI / Ollama"]
    end

    root[("ORACLE_SCAN_ROOT<br/>local git repos")] --> scan
    embed --> store[("store/sqlite-store.ts<br/>sqlite-vec")]

    mcp --> retr["retrieval/chain.ts"]
    cli --> retr
    retr --> store
    retr --> answer["cited chunks / LLM answer"]
```

Markdown files with a leading YAML frontmatter block get `fmType` / `fmTitle` / `fmTags` / `fmSources` chunk metadata alongside the usual `repo` / `filePath` / `lineStart` / `lineEnd` fields. `oracle_search` can filter on this metadata (`type`, `tags`) and `oracle_query` surfaces a `Pointers` section built from it automatically; see [docs/architecture.md](docs/architecture.md#chunking) and [docs/mcp.md](docs/mcp.md#okf-frontmatter-filters-type--tags) for details.

## Install

From npm (recommended for MCP-only use):

```bash
npm i -g @lannguyensi/codebase-oracle
```

This puts a `codebase-oracle` binary on your PATH. Use it as a CLI or as the entry for an MCP client.

From source (for development, or to run `npm run index` over a custom scan root):

```bash
git clone https://github.com/LanNguyenSi/codebase-oracle.git
cd codebase-oracle
npm install && npm run build
```

## Try it in 60 seconds

```bash
# point at the directory holding your git repos, set your key
export ORACLE_SCAN_ROOT=~/code
export OPENAI_API_KEY=sk-...

# build the index, then ask a question
codebase-oracle index
codebase-oracle query "where do we handle auth?"
```

Or wire it into Claude Code as an MCP server:

```bash
claude mcp add codebase-oracle -- codebase-oracle mcp
```

From any Claude Code session on the same machine you can now call `oracle_search`, `oracle_query`, `oracle_expand`, `oracle_list_repos`, and `oracle_reindex` against the shared index. `oracle_reindex` triggers an incremental re-index on demand (only changed and new files are re-embedded); use it after merging code you want the oracle to see immediately, instead of waiting for the next scheduled scan.

## What a search looks like

`oracle_search` with `query="where do we read AGENT_TASKS_TOKEN"` returns matching chunks with line-number locations:

```
[1] src/auth/token.ts:14-32 (agent-tasks-cli):
function loadToken(): string {
  const value = process.env.AGENT_TASKS_TOKEN;
  if (!value) throw new Error("AGENT_TASKS_TOKEN missing");
  return value;
}

---

[2] backend/src/middleware/auth.ts:8-21 (agent-tasks):
export function requireToken(req, res, next) {
  const token = req.headers.authorization?.replace(/^Bearer /, "");
  if (token !== process.env.AGENT_TASKS_TOKEN) return res.sendStatus(401);
  next();
}
```

Chunks from markdown files with OKF frontmatter metadata add a `[type]` tag to the header and, when present, a `sources:` line:

```
[3] docs/okf/backend.md:1-40 (agent-tasks) [module]:
sources: agent-tasks/backend/src/config.ts, agent-tasks/backend/src/server.ts
...
```

`oracle_search --type module` matches only chunks whose frontmatter `type` field strictly equals `module`; `--tags okf,backend` matches chunks whose frontmatter `tags` contain **all** of the listed tags. Both filters only match chunks that HAVE the field: chunks without frontmatter metadata are excluded whenever `type` or `tags` is set, and both AND-compose with `--repo` / `--path-glob`.

`oracle_query` answers get an automatic `Pointers (from OKF sources metadata):` section appended after the sources list when any retrieved chunk carries `fmSources`, listing the deduped union of paths in retrieval-rank order (capped at 10, with a truncation note past that). No LLM involvement, and the section is omitted entirely when nothing in the retrieved context has `fmSources`.

`oracle_list_repos` shows what's indexed and how fresh each repo is:

```
- agent-tasks — 1842 chunks across 287 files (indexed 2026-04-27T10:14:02Z, 14 min ago)
- agent-tasks-cli — 421 chunks across 68 files (indexed 2026-04-27T10:14:18Z, 14 min ago)
```

## Next steps

| If you want to... | Read |
|------|------|
| Wire it into Claude Code (MCP setup, the five tools, HTTP MCP auth) | [docs/mcp.md](docs/mcp.md) |
| Switch to Ollama, change embedding models, customise scan filters | [docs/configuration.md](docs/configuration.md) |
| Understand how the index is built (chunking, embeddings, sqlite-vec) | [docs/architecture.md](docs/architecture.md) |
| Migrate from v0.2 (JSONL) or pick up v0.4 line numbers | [docs/upgrades.md](docs/upgrades.md) |

## CLI reference

The CLI auto-loads `.env` from the current working directory if present (the repo root when run via the `npm run` scripts below).

```bash
npm run index                            # build/refresh the index over ORACLE_SCAN_ROOT
npm run index -- --path /path/to/repos   # custom scan root
npm run query -- "what is the audit system?"
npm run query -- -r my-repo "where is the schema defined?"
npm run query -- -k 20 "list all API endpoints"
npm run dev -- search "evaluateTransitionRules"
npm run dev -- search "okf backend" --type module --tags okf,backend
npm run dev -- list-repos                # indexed repos with chunk/file counts + freshness
npm run dev -- expand my-repo path/to/file.ts -l 42   # read a window of lines around a position
npm run watch                            # keep the index fresh in the background
npm run migrate-store                    # migrate a v0.2.0 embeddings.jsonl to the SQLite store
```

| Flag | Description |
|------|-------------|
| `-r, --repo <name>` | Filter results to a specific repo |
| `-k, --limit <n>` | Number of chunks to retrieve (default: 12 for `query`, 10 for `search`) |
| `-g, --path-glob <glob>` | (`search` only) Filter results by file path glob (e.g. `**/.github/workflows/*.yml`) |
| `-t, --type <type>` | (`search` only) Filter to chunks whose `fmType` OKF frontmatter metadata strictly equals this value. Excludes chunks without frontmatter metadata. |
| `--tags <tags>` | (`search` only) Comma-separated; filter to chunks whose `fmTags` OKF frontmatter metadata contains ALL listed tags. Excludes chunks without frontmatter metadata. |
| `--no-expand-sources` | (`search` only) Disable OKF sources-expansion (do not inject files pointed at by a retrieved doc's `sources:` frontmatter); expansion is on by default. |

Watch mode runs a chokidar watcher over the scan root and re-embeds changed files after a short debounce. Newly dropped `.git` roots need one explicit `npm run index` to back-fill before watch mode picks up subsequent edits. See [docs/architecture.md](docs/architecture.md#watch-mode) for details.

## Two use cases

**For agents (primary).** A local Claude Code or other MCP client talks to the oracle's MCP server over stdio. The agent runs `oracle_search` / `oracle_query` / `oracle_expand` / `oracle_list_repos` / `oracle_reindex` against a shared, pre-built index: it never has to scan the filesystem, embed anything, or burn its own context on grep output. One scan for everyone, semantic instead of regex, no duplicate embeddings, MCP-first design.

**For humans.** The CLI is useful for spot checks, debugging the index, or terminal-driven answers without going through an agent.

## Development

```bash
npm run build          # TypeScript compilation
npm test               # vitest run
npx tsc --noEmit       # type check only
```

## Releasing

Retrieval quality is guarded by a hand-labelled eval set rather than by CI. The
eval needs an embedding provider (`OPENAI_API_KEY`, or an OpenAI-compatible
endpoint such as Ollama) and costs under a cent per run, so it runs as a
**manual pre-release gate**, not on every PR:

```bash
npm run eval           # compares retrieval against tests/eval/baseline.json
```

Run it before tagging a release and paste the final line into the release PR. A
regression vs. baseline blocks the release until the cause is fixed or the
baseline is updated with a documented reason. See
[tests/eval/README.md](tests/eval/README.md) for the full workflow, including
how to add questions and corpus repos.

## License

MIT. See [docs/architecture.md#credits](docs/architecture.md#credits) for inspiration and prior art.
