# codebase-oracle

Semantic search across all your local repos, via MCP or CLI.

[![CI](https://github.com/LanNguyenSi/codebase-oracle/actions/workflows/ci.yml/badge.svg)](https://github.com/LanNguyenSi/codebase-oracle/actions/workflows/ci.yml)

codebase-oracle builds one semantic index over every git repo under a root directory, then exposes it to agents via MCP or to humans via CLI. The vector store lives on your machine; embeddings are computed by OpenAI by default, or fully local via Ollama (configurable). Indexing is incremental: only new and changed files are re-embedded. Built for agents first, humans second.

## Key features

- One incremental index over every git repo under `ORACLE_SCAN_ROOT`, shared by the CLI and the MCP server.
- MCP server (5 tools over stdio, 4 over HTTP) so agents query a pre-built index without scanning or embedding anything themselves.
- OpenAI or local Ollama embeddings; Anthropic, OpenAI, or any OpenAI-compatible endpoint for answers.
- Answers are grounded: retrieved chunks carry `path:line_start-line_end (repo)` locations and are cited in generated answers.
- OKF frontmatter awareness: `type`/`tags` search filters and an automatic `Pointers` section built from a doc's `sources:` metadata.
- `--json` output on the query, search, list-repos, and expand commands, with a documented success/failure/degraded contract for scripting.

## Quick start

Prerequisites: Node.js 22+, and an embedding provider (an OpenAI API key by default, or a local Ollama instance with `ORACLE_EMBEDDING_PROVIDER=ollama`; see [docs/configuration.md](docs/configuration.md)).

```bash
npm i -g @lannguyensi/codebase-oracle
```

This puts a `codebase-oracle` binary on your PATH, usable as a CLI or as an MCP server entry point. From source instead (for development, or `npm run index` over a custom scan root):

```bash
git clone https://github.com/LanNguyenSi/codebase-oracle.git
cd codebase-oracle
npm install && npm run build
```

Then point it at your repos and build the index:

```bash
export ORACLE_SCAN_ROOT=~/code
export OPENAI_API_KEY=sk-...

codebase-oracle index
codebase-oracle query "where do we handle auth?"
```

Or wire it into Claude Code as an MCP server:

```bash
claude mcp add codebase-oracle -- codebase-oracle mcp
```

From any Claude Code session on the same machine you can now call `oracle_search`, `oracle_query`, `oracle_expand`, `oracle_list_repos`, and `oracle_reindex` against the shared index. See [docs/mcp.md](docs/mcp.md) for registration recipes and the full tool surface.

## Usage

`oracle_search` (also available as `codebase-oracle search` / `npm run dev -- search`) returns matching chunks with line-number locations:

```bash
codebase-oracle search "where do we read AGENT_TASKS_TOKEN"
```

```
[1] src/auth/token.ts:14-32 (my-repo):
function loadToken(): string {
  const value = process.env.AGENT_TASKS_TOKEN;
  if (!value) throw new Error("AGENT_TASKS_TOKEN missing");
  return value;
}
```

`oracle_query` asks a natural-language question and returns an LLM answer with citations instead of raw chunks. Every answer LLM call is bounded by `ORACLE_LLM_TIMEOUT_MS` (default 120000ms): an unreachable or unresponsive endpoint falls back to raw retrieved context within that bound instead of hanging; see [docs/configuration.md](docs/configuration.md) for the full reasoning and env-var reference. Full CLI flags, `--json` output shapes, and more examples: [docs/cli-reference.md](docs/cli-reference.md).

## Documentation

| If you want to... | Read |
|------|------|
| Wire it into Claude Code (MCP setup, the five tools, HTTP MCP auth) | [docs/mcp.md](docs/mcp.md) |
| Switch to Ollama, change embedding models, customise scan filters | [docs/configuration.md](docs/configuration.md) |
| Understand how the index is built (chunking, embeddings, sqlite-vec) | [docs/architecture.md](docs/architecture.md) |
| Full CLI reference (flags, `--json` contract) | [docs/cli-reference.md](docs/cli-reference.md) |
| Migrate from v0.2 (JSONL) or pick up v0.4 line numbers | [docs/upgrades.md](docs/upgrades.md) |

## Development and contributing

```bash
npm run build          # TypeScript compilation
npm test               # vitest run
npx tsc --noEmit       # type check only
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the PR workflow, dev setup, and the release process.

## License

MIT. See [docs/architecture.md#credits](docs/architecture.md#credits) for inspiration and prior art.
