---
type: invariant
title: Ingest skips are loud, and enforced in two independent places
description: Oversize/read-error skips are reported (never swallowed) while empty files skip silently; the stat-first size gate is reimplemented separately in scanner.ts and watch.ts, so both must change together.
tags: [ingest, scanner, watch, skips, config]
timestamp: 2026-07-10T08:34:41.256341Z
sources:
  - src/config.ts
  - src/ingest/scanner.ts
  - src/ingest/runner.ts
  - src/mcp-server.ts
  - src/watch.ts
  - CHANGELOG.md
---

# Ingest skips are loud, and enforced in two independent places

## Invariant

A file that ingest declines to index for **size** or a **read error** must be
reported (one `WARNING:` line per file), never silently dropped. A file skipped
for being **empty** is skipped silently on purpose. This size/empty gate is
implemented **twice** — once in `scanner.ts` (full index) and once in
`watch.ts` (incremental re-embed) — as two separate code paths reading the same
config. A behaviour change in one does not propagate to the other; both must be
edited together.

## Config: the size ceiling and its fail-loud parse

`src/config.ts`:
- `maxFileSizeBytes: z.number().int().positive().default(500_000)` (config.ts:63).
  Positive int required; the default is 500 KB.
- Env plumbing: `maxFileSizeBytes: parseMaxFileSizeBytes(process.env.ORACLE_MAX_FILE_SIZE)` (config.ts:115).
- `parseMaxFileSizeBytes(raw)` (config.ts:145-148): `undefined` or `raw.trim() === ""`
  returns `undefined` (schema default applies); otherwise `Number(raw)` is handed
  straight to `configSchema.parse`, where `.int().positive()` rejects `NaN`/`0`/negative
  and `loadConfig` throws.
- The empty-`.env`-line contract is spelled out in the comment at config.ts:137-144:
  an `ORACLE_MAX_FILE_SIZE=` line "must not crash the CLI" (treated as unset), but
  "A typo'd ORACLE_MAX_FILE_SIZE must fail loudly, not silently resolve to some other limit."

## Scanner: three skip classes, only two reported

`src/ingest/scanner.ts`, `walkRepo` → inner `walk`:
- **Stat before read** (scanner.ts:140-151): `const st = await stat(fullPath);`
  then `if (st.size > maxFileSizeBytes)` calls `onSkip({... reason: "too-large", sizeBytes, limitBytes})`
  and `continue` — the size decision is made in true bytes, before the file is read
  into memory (replacing the old in-memory `content.length > 200_000` char count).
- **Empty files skip SILENTLY** (scanner.ts:153-157): after `readFile`, `if (!content.trim()) continue;`
  with no `onSkip`. Comment: "silent skip on purpose — unlike too-large/read-error, it is not an anomaly worth a WARNING line."
- **Read errors are reported, not swallowed** (scanner.ts:167-179): the `catch (err)`
  calls `onSkip({... reason: "read-error", message})`. Non-throwing (one bad file
  must not kill the scan) but never a bare empty catch.
- `SkippedFile.reason` is `"too-large" | "read-error"` (scanner.ts:73) — the empty
  case has no reason value because it is never surfaced.
- `onSkip` defaults to a no-op (scanner.ts:98); the indexer and watch pass a real one.
- **Asymmetry:** three skip classes (too-large, read-error, empty); only two are
  ever reported.

## Runner: skip count flows into the summary and the WARNING lines

`src/ingest/runner.ts`:
- `IndexSummary` carries `filesSkipped: number` and `skippedFiles: Array<{repo, relativePath, reason, sizeBytes?, limitBytes?, message?}>` (runner.ts:29-37).
- `walkOptions.onSkip` pushes each skip into `skippedFiles` (runner.ts:93-102).
- **Loud per-file + total reporting** (runner.ts:165-178): for each skip, `warn(...)`
  emits `WARNING: skipped <path> — <bytes> bytes > ORACLE_MAX_FILE_SIZE=<limit>`
  (too-large) or `WARNING: skipped <path> — read error: <message>`, then a one-line
  total `WARNING: N file(s) skipped during scan; raise ORACLE_MAX_FILE_SIZE...`.
- The `warn` sink defaults to a no-op (runner.ts:66); the CLI routes it to stderr,
  MCP and zero-skip runs stay quiet.
- `formatIndexSummary` folds the count into the **files** segment (runner.ts:334-345):
  `... ${summary.filesNew} new, ${summary.filesPruned} pruned, ${summary.filesSkipped} skipped).`

## MCP: only the count reaches oracle_reindex, not the per-file lines

`src/mcp-server.ts` oracle_reindex handler (mcp-server.ts:260-263):
`const summary = await runIndex(cfg);` then returns `formatIndexSummary(summary)`.
- `runIndex` is called **without** a `warn` sink, so the per-file `WARNING:` lines
  are suppressed on the MCP path (they only appear on CLI stderr).
- The **skip count** still reaches the MCP reindex summary via
  `formatIndexSummary`'s files segment. So: count → MCP; per-file lines → CLI only.

## The load-bearing duplication (watch.ts is a separate enforcement site)

`src/watch.ts` does **not** call `scanner.ts`. `loadScannedFile` (watch.ts:121-154)
independently reimplements the stat-first gate:
- `const limit = maxFileSizeBytes ?? DEFAULT_MAX_FILE_SIZE_BYTES;` then
  `const st = await stat(absolutePath); if (st.size > limit) return {kind:"too-large", ...}` (watch.ts:132-139).
- Empty check `if (!content.trim()) return {kind:"empty"}` (watch.ts:141).
- It is fed the same config value: the caller passes `config.maxFileSizeBytes`
  into `loadScannedFile(...)` (watch.ts:261).

**watch reports too-large loudly too — verified.** In `flush`, `loaded.kind === "too-large"`
emits `console.warn("WARNING: skipped <path> — <bytes> bytes > ORACLE_MAX_FILE_SIZE=<limit>")`
and unindexes any stale vectors for that file (watch.ts:271-281). `loaded.kind === "empty"`
is a **silent** skip (watch.ts:283-292, mirroring scanner.ts), though it still clears
stale vectors. Note watch reports via `console.warn` directly, whereas runner routes
through an injected `warn` sink — different mechanisms, same "loud" outcome.

**Consequence:** the size/empty gate lives in two places (`scanner.walkRepo`
scanner.ts:140-157 and `watch.loadScannedFile` watch.ts:132-141). A fix or
behaviour change to one does **not** automatically apply to the other. Any change
to the threshold semantics, the empty-file rule, or the reporting shape must be
made in **both** files, or full-index and watch-mode behaviour will silently diverge.

## Why the reporting is loud (historical)

Per `CHANGELOG.md` `[0.10.0]` (2026-07-04): `agent-tasks/backend/src/routes/tasks.ts`
(207,716 bytes) "was silently dropped by the old `content.length > 200_000` check;
it and any file like it are now reported, not swallowed" (CHANGELOG.md:30). The same
release moved the limit to a `stat`-first byte check "in both the scanner and
`watch.ts`, which previously duplicated the old limit as its own `MAX_FILE_BYTES`
constant" (CHANGELOG.md:26) — i.e. the duplication predates the fix and was carried
forward, not introduced by it.
```

---

Verification notes for the caller:
- Every lead claim held; no DISCREPANCIES section needed.
- One precision worth flagging: on the MCP path, `runIndex(cfg)` at `src/mcp-server.ts:260` is called with no `warn` sink, so the per-file `WARNING:` lines are NOT emitted to MCP — only the folded skip count (via `formatIndexSummary`) reaches the reindex summary. The doc states this explicitly.
- watch.ts reports oversize skips via `console.warn` directly (watch.ts:271-275), not through the injected `warn` sink that runner.ts uses — same "loud" outcome, different mechanism. Captured in the doc.
- CHANGELOG confirms both the 207,716-byte `agent-tasks/backend/src/routes/tasks.ts` incident (line 30) and that watch.ts previously carried its own duplicate `MAX_FILE_BYTES` constant (line 26).
