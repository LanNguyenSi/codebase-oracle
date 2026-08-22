---
type: overview
title: MCP tools — where the surface is documented
description: Pointer doc. ../mcp.md is the authoritative tool reference (five stdio tools, the HTTP parity gap, the type/tags/expand_sources parameters); this entry points to sources-expansion.md for expand_sources semantics.
tags: [overview, mcp, tools, pointer]
timestamp: 2026-08-22T04:51:51Z
sources:
  - docs/mcp.md
  - src/mcp-server.ts
---

# MCP tools — pointer

[../mcp.md](../mcp.md) is authoritative for the MCP tool surface: the stdio
tools, the documented HTTP parity gap (HTTP exposes four of five tools and no
`path_glob`), and the `type` / `tags` frontmatter filters on `oracle_search`.
Not duplicated here.

`oracle_search` also accepts an **`expand_sources`** parameter
(`src/mcp-server.ts`), documented in `mcp.md`. What it does, and the dedup
and cap semantics that make its effect non-obvious, are in
[sources-expansion.md](sources-expansion.md).
