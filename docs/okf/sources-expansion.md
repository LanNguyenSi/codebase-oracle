---
type: invariant
title: Sources-expansion — how fmSources become retrievable chunks
description: oracle_search injects the first chunk of each file an organic hit's OKF fmSources points at, parent-namespace-first, deduped by (repo,filePath); since 0.10.2 a below-parent organic hit is hoisted into the injection slot instead of skipped, capped to limit. The expand_sources parameter is listed in docs/cli-reference.md and mcp.md; the dedup/hoist/per-parent-cap semantics live only here and in code.
tags: [okf, sources-expansion, retrieval, search]
timestamp: 2026-09-26T04:55:48Z
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
(`src/store/sqlite-store.ts:371-372#"ORDER BY rowid LIMIT 1"`, exposed via
`getFirstChunkByFileInternal` at
`src/store/sqlite-store.ts:777-784#"if (!row) return null;"`). At most
`MAX_INJECTIONS_PER_PARENT = 3` chunks are injected per parent
(`src/retrieval/chain.ts:583#"MAX_INJECTIONS_PER_PARENT = 3"`), while at most
`MAX_SOURCES_EXAMINED_PER_PARENT = 20` raw `fmSources` entries are even looked
at per parent (`src/retrieval/chain.ts:591#"MAX_SOURCES_EXAMINED_PER_PARENT = 20"`), the
second bound caps synchronous store lookups against an adversarial doc that
lists thousands of sources.

The `expand_sources` parameter itself is documented: `docs/cli-reference.md`'s
flag table lists `--no-expand-sources`, and `docs/mcp.md` lists `expand_sources` in
the tool table and the stdio tool-inputs summary. What is documented only here
and in code is the semantics: the parent-namespace-first resolution, the
`(repo, filePath)` dedup with organic-wins, the hoist behavior, and the
per-parent injection and examined-sources caps.

Each synthesized-injection `Document` is minted fresh with a transient
`expandedFrom` marker carrying the parent's `filePath`: `metadata: { ...chunk.metadata,
expandedFrom: parentFilePath }` (`src/retrieval/chain.ts:766#"expandedFrom: parentFilePath"`). A hoisted row
(see below) carries no such marker: it is the existing organic `Document`, moved,
not a new one. The marker is render-only:
`formatChunkExpandedTag` (`src/retrieval/chain.ts:88#"function formatChunkExpandedTag("`, called at
`src/retrieval/chain.ts:121#"formatChunkExpandedTag(metadata)"`) turns it
into an `[expanded from <basename>]` tag in the search output, spliced alongside
the `[type]` tag. Injected Documents are **never persisted**; they
exist only in the returned list for that one call.

**Byte-identical-to-off guarantee.** `expandSourcesInResults` pushes each
parent, then either hoists an organic pointed-at file into place or resolves a
synthesized injection for one with no organic hit anywhere. On the synthesized
path, a non-matching entry (directory, glob, typo, absent file) resolves to
`null` and is skipped silently and deterministically
(`src/retrieval/chain.ts:749-759#"if (!chunk) continue;"`). When no row
carries a resolvable `fmSources` entry, the returned list is the organic list
unchanged, identical to `expandSources: false`. `expandSources` defaults to
`true` (`searchCodebase`, `src/retrieval/chain.ts:782#"options?.expandSources ?? true"`).

## Where it's enforced

`expandSourcesInResults(organic, vectorStore, limit)` at
`src/retrieval/chain.ts:669-773#"return expanded.slice(0, limit);"` is the whole mechanism. It runs after ranking,
inside `searchCodebase`.

**Path shape (load-bearing).** `fmSources` entries are **repo-root-relative** by
OKF convention (e.g. `backend/src/app.ts`). The store's `file_path` namespace
depends on scan layout: when repos are subdirectories of the scan root (the
common production case), stored paths carry the **repo prefix** (e.g.
`agent-tasks/backend/src/app.ts`). The candidate-order logic now lives in its
own helper, `resolveSourcePathCandidates` (`src/retrieval/chain.ts:609-618#"return [first, second];"`),
extracted in 0.10.2 so the store lookup and the organic-hit hoist check share
one definition of which path form a given `fmSources` entry means.
`resolveSourceChunk` (`src/retrieval/chain.ts:623-636#"getFirstChunkByFile(repo, second)"`) then tries the store
lookup against both forms in that order. Both derive the namespace from
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
(`src/mcp-server.ts:133-137#"(default true)"`, threaded to `searchCodebase` as `expandSources`
at `src/mcp-server.ts:148#"expandSources: expand_sources"`), described as "inject files pointed at by a retrieved doc's OKF
sources: frontmatter, marked [expanded from ...] (default true)". `docs/mcp.md`
documents `expand_sources` too.

**Not to be confused with `oracle_expand`.** That is a separate, unrelated MCP
tool (`src/mcp-server.ts:189#"oracle_expand"`) that reads a window of lines around a position in
an indexed file. The `expand_sources` parameter described here is a flag on
`oracle_search`. Similar names, unrelated mechanisms.

## What the dedup rule actually does now

**Dedup + hoist — fixed in 0.10.2, not a known limitation anymore.** The dedup
key is `(repo, filePath)` via `fileKeyOf`
(`src/retrieval/chain.ts:593-597#"::${filePath}"`). Every organic hit is
indexed into an `organicByKey` map, first occurrence wins
(`src/retrieval/chain.ts:677-680#"organicByKey.set(key, doc)"`), and a separate
`placed` set tracks every row, organic push, hoist, or synthesized injection,
that has actually landed in the output
(`src/retrieval/chain.ts:687#"const placed = new Set"`). A pointed-at file that is an organic hit ranked **at or above** its
pointing parent (already placed by the time the parent's own `fmSources` are
processed) is left untouched: no duplicate, no reorder. A pointed-at file that
is an organic hit ranked **below** its pointing parent (not yet placed,
whether or not it would have survived the `limit` cut on its own) is
**hoisted**: its existing organic `Document` (real chunk, real snippet) is
moved into the injection slot right after the parent, with no `expandedFrom`
marker, instead of being left at its natural rank where a lower-priority
sibling injection could push it past the cut
(`src/retrieval/chain.ts:737-743#"expanded.push(organicByKey.get(hoistKey)!)"`).
Only a file with **no organic hit anywhere** in the candidate list falls back to
a synthesized first-chunk injection tagged `expandedFrom`
(`src/retrieval/chain.ts:749-766#"expandedFrom: parentFilePath"`). The combined
parent+injection+hoist list is still capped with `.slice(0, limit)`
(`src/retrieval/chain.ts:773#"expanded.slice(0, limit)"`), so a hoist is not
exempt from the final cut either.

Consequences:

- "Hoist," not "displace," is now the code's own vocabulary: the word
  appears throughout `expandSourcesInResults` and its comments (e.g.
  `src/retrieval/chain.ts:653#"HOISTED"`, `src/retrieval/chain.ts:657#"A hoisted row"`,
  `src/retrieval/chain.ts:675#"be hoisted"`,
  `src/retrieval/chain.ts:683-686#"already-hoisted"`,
  `src/retrieval/chain.ts:731-745#"injected++;"`); "displace" survives only
  once, in a comment about the exact failure mode this fix closes
  (`src/retrieval/chain.ts:656#"displace"`).
- A pointed-at file that is organically present below the cut is now
  **promoted** into the injection slot instead of silently left to be sliced
  away, which was the original regression this mechanism used to have.
- Once `expanded.length >= limit`, the loop still breaks early
  (`src/retrieval/chain.ts:704-706#"if (expanded.length >= limit) break;"`),
  skipping later parents and their store lookups entirely.

Fixed in codebase-oracle 0.10.2 (`CHANGELOG.md:#0.10.2`,
commit `5b59ec6`, "fix(search): hoist below-cut organic hits instead of
skipping their injection"). Describe the mechanism in hoist terms: a
below-parent organic hit is hoisted into place, not silently skipped.

**Do not confuse with `oracle_query`'s pointers section.** `oracle_query`
appends a text block titled `"Pointers (from OKF sources metadata):"`
(`POINTERS_SECTION_LABEL`, `src/retrieval/chain.ts:167#"POINTERS_SECTION_LABEL ="`) built by
`extractSourcePointers` (`src/retrieval/chain.ts:151-164#"return pointers;"`) and
`formatPointersSection` (`src/retrieval/chain.ts:173-180#"${POINTERS_SECTION_LABEL}"`,
`POINTERS_CAP = 10`). That is **plain text listing source paths after an LLM
answer**: it renders `fmSources` strings, it does not retrieve anything.
Sources-expansion instead injects **retrievable chunks** into the
`oracle_search` result list. Two distinct mechanisms, frequently conflated: one
prints pointer strings, the other pulls in actual file content.

**Things that silently break injection:** parent metadata missing `repo`
(`parentRepo.length === 0` → skip,
`src/retrieval/chain.ts:714#"if (parentRepo.length === 0) continue;"`);
`fmSources` not an array
(`src/retrieval/chain.ts:710#"if (!Array.isArray(fmSources)) continue;"`); a
source string that is empty or non-string
(`src/retrieval/chain.ts:722#"src.length === 0) continue;"`); a source that
resolves to no stored chunk under either path shape
(`src/retrieval/chain.ts:759#"if (!chunk) continue;"`); the per-parent cap of 3
or the per-parent examination cap of 20 being hit
(`src/retrieval/chain.ts:721#"if (injected >= MAX_INJECTIONS_PER_PARENT) break;"`,
`src/retrieval/chain.ts:719#"if (examined >= MAX_SOURCES_EXAMINED_PER_PARENT) break;"`);
or the global `limit` already being reached
(`src/retrieval/chain.ts:706#"if (expanded.length >= limit) break;"`).
