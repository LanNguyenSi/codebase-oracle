---
type: invariant
title: Sources-expansion — how fmSources become retrievable chunks
description: oracle_search injects the first chunk of each file an organic hit's OKF fmSources points at, parent-namespace-first, deduped by (repo,filePath) with organic-wins, capped to limit — the repo's one undocumented feature.
tags: [okf, sources-expansion, retrieval, search]
timestamp: 2026-07-10T08:34:41.256341Z
sources:
  - src/retrieval/chain.ts
  - src/store/sqlite-store.ts
  - src/mcp-server.ts
---

## The invariant

When `oracle_search` returns results, every organic hit whose OKF frontmatter
carries an `fmSources` array causes a **representative chunk** of each file that
array points at to be injected into the result list, immediately after its
parent, in `fmSources` order. "Representative chunk" is concretely the file's
**first chunk**: `getFirstChunkByFile` runs `SELECT page_content, metadata FROM
docs WHERE repo = ? AND file_path = ? ORDER BY rowid LIMIT 1`
(`src/store/sqlite-store.ts:293-295`, exposed via
`getFirstChunkByFileInternal` at `:638-650`). At most
`MAX_INJECTIONS_PER_PARENT = 3` chunks are injected per parent
(`src/retrieval/chain.ts:512`), while at most `MAX_SOURCES_EXAMINED_PER_PARENT
= 20` raw `fmSources` entries are even looked at per parent (`:520`) — the
second bound caps synchronous store lookups against an adversarial doc that
lists thousands of sources.

This feature is **undocumented**. Grepping `docs/` and `README.md` for
`expandSources`, `expand_sources`, or `sources-expansion` returns zero hits.
It exists only in code and code comments.

Each injected `Document` is minted fresh with a transient `expandedFrom` marker
carrying the parent's `filePath`: `metadata: { ...chunk.metadata, expandedFrom:
parentFilePath }` (`src/retrieval/chain.ts:612`). The marker is render-only —
`renderExpandedFromTag` turns it into a `[expanded from <basename>]` suffix on
the search output (`:75-87`). Injected Documents are **never persisted**; they
exist only in the returned list for that one call.

**Byte-identical-to-off guarantee.** `expandSourcesInResults` pushes each parent
and only appends injections when a source actually resolves to a stored chunk. A
non-matching entry (directory, glob, typo, absent file) resolves to `null` and
is skipped silently and deterministically (`:597-605`). When no row carries a
resolvable `fmSources` entry, the returned list is the organic list unchanged —
identical to `expandSources: false`. `expandSources` defaults to `true`
(`searchCodebase`, `src/retrieval/chain.ts:628`).

## Where it's enforced

`expandSourcesInResults(organic, vectorStore, limit)` at
`src/retrieval/chain.ts:564-620` is the whole mechanism. It runs after ranking,
inside `searchCodebase`.

**Path shape (load-bearing).** `fmSources` entries are **repo-root-relative** by
OKF convention (e.g. `backend/src/app.ts`). The store's `file_path` namespace
depends on scan layout: when repos are subdirectories of the scan root (the
common production case), stored paths carry the **repo prefix** (e.g.
`agent-tasks/backend/src/app.ts`). `resolveSourceChunk`
(`src/retrieval/chain.ts:537-551`) bridges this by deriving the namespace from
the **parent chunk's own path** — parent-namespace-first:

```
const prefixed = `${repo}/${src}`;
const parentIsPrefixed = parentFilePath.startsWith(`${repo}/`);
const first  = parentIsPrefixed ? prefixed : src;   // try parent's shape first
const second = parentIsPrefixed ? src : prefixed;   // then the other form
return getFirstChunkByFile(repo, first) ?? getFirstChunkByFile(repo, second);
```

So for repo `agent-tasks`, parent `agent-tasks/backend/src/app.ts`, and source
`backend/src/db.ts`, the lookup tries `agent-tasks/backend/src/db.ts` first,
then bare `backend/src/db.ts`. If the parent's `filePath` were unprefixed the
order flips. This parent-derived resolution is why a fixture using the wrong
path shape once passed while doing nothing in production: raw-only lookup
silently no-opped on every real repo while fixture-shaped tests passed (fixed in
commit `a3f48d3`, "fix(search): resolve fmSources against the store's
repo-prefixed path namespace", verified present via `git log`).

**MCP surface.** `oracle_search` exposes the `expand_sources` boolean param
(`src/mcp-server.ts:133-138`, threaded to `searchCodebase` as `expandSources`
at `:148`), described as "inject files pointed at by a retrieved doc's OKF
sources: frontmatter, marked [expanded from ...] (default true)". `docs/mcp.md`
does **not** document `expand_sources` (grep confirms absence).

## What breaks it

**Dedup + cap — current behaviour and a known limitation, not a desirable
invariant.** The dedup key is `(repo, filePath)` via `fileKeyOf`
(`src/retrieval/chain.ts:522-527`). The `seen` set is seeded with **every
organic hit** before any injection (`:569-573`), so a file already retrieved
organically is never injected — "organic wins" — **even if that organic hit
sits below the final `limit` cut**. The combined parent+injection list is then
returned as `.slice(0, limit)` (`:619`).

Consequences:

- Injections push out **tail organic rows**: the slice is applied after
  injections are interleaved, so injected chunks can displace organic rows that
  ranked below them. (The word "displace" appears once in this repo, in the code comment at
  `:560`; "hoist" appears nowhere. Both are otherwise task-tracker vocabulary.)
- A pointed-at file that is **organically present below the cut** is neither
  injected (it is in `seen`) nor promoted (there is no reordering) — it stays
  below the cut and may be sliced away entirely, while a **non-target sibling**
  gets pushed out instead.
- Once `expanded.length >= limit`, the loop breaks early (`:578-580`),
  skipping later parents and their store lookups entirely.

Tracked as agent-tasks `d165ff85`. Describe the mechanism in these terms
(seed-all-organic + `.slice(0, limit)`), not as "hoist/displace" — those are the
tracker's words.

**Do not confuse with `oracle_query`'s pointers section.** `oracle_query`
appends a text block titled `"Pointers (from OKF sources metadata):"`
(`POINTERS_SECTION_LABEL`, `src/retrieval/chain.ts:160`) built by
`extractSourcePointers` (`:144-158`) and `formatPointersSection` (`:166-174`,
`POINTERS_CAP = 10`). That is **plain text listing source paths after an LLM
answer** — it renders `fmSources` strings, it does not retrieve anything.
Sources-expansion instead injects **retrievable chunks** into the
`oracle_search` result list. Two distinct mechanisms, frequently conflated: one
prints pointer strings, the other pulls in actual file content.

**Things that silently break injection:** parent metadata missing `repo`
(`parentRepo.length === 0` → skip, `:588`); `fmSources` not an array
(`:584`); a source string that is empty or non-string (`:596`); a source that
resolves to no stored chunk under either path shape (`:605`); the per-parent
cap of 3 or the per-parent examination cap of 20 being hit (`:593`, `:595`);
or the global `limit` already being reached (`:580`).
