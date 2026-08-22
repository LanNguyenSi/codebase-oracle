# Knowledge bundle index

Curated OKF knowledge bundle for codebase-oracle. The `docs/` tree one level up
is already accurate and current for architecture, the MCP tool surface, and
configuration; those get pointer entries here rather than copies. The concept
docs below cover semantics that live only in code, or that no doc states.

Note: this repo indexes its own `docs/` tree, so these files are retrievable
through the very tool they describe.

## Overview

- [Architecture](architecture-pointer.md), pointer to `../architecture.md` plus
  what it does not cover.
- [MCP tools](mcp-pointer.md), pointer to `../mcp.md` for the full tool
  surface.
- [Configuration](configuration-pointer.md), pointer to `../configuration.md`
  plus the token-budget asymmetry it does not mention.

## Invariants

- [Sources-expansion](sources-expansion.md), how an organic hit's `fmSources`
  become injected, retrievable chunks: first chunk, parent-namespace-first path
  resolution, `(repo, filePath)` dedup with organic-wins, and the `limit` cap.
  The parameter is documented in README.md and mcp.md; the semantics live
  only here and in code.
- [Ingest size limit](ingest-size-limit-enforcement.md), oversize and read-error
  skips are loud while empty-file skips are silent, and the gate is implemented
  independently in `scanner.ts` and `watch.ts`.
- [Provider enums and token budget](provider-enums-and-token-budget.md), two
  separate provider enums, and why only the Anthropic lane caps `maxTokens`
  (the blank-answer failure mode on thinking models).

## Runbooks

- [Index-data freshness vs server-code freshness](index-freshness-vs-code-freshness.md),
  two independent staleness axes with different fixes: reindexed data is visible
  to a running server without restart; changed source is not.
