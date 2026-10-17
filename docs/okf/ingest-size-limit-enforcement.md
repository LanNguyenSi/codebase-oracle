---
type: invariant
title: Ingest skips are loud, and enforced in two independent places
description: Oversize/read-error skips are reported (never swallowed) while empty files skip silently; the stat-first size gate is reimplemented separately in scanner.ts and watch.ts, so both must change together; the `.codebase-oracle-skip` sentinel check is shared. Since the per-type ceiling was added, the applicable env var name travels with each skip so the WARNING names the knob that actually needs raising.
tags: [ingest, scanner, watch, skips, config]
timestamp: 2026-10-17T07:32:59Z
sources:
  - src/config.ts
  - src/ingest/scanner.ts
  - src/ingest/runner.ts
  - src/mcp-server.ts
  - src/watch.ts
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

## Two independent size ceilings, chosen per file extension

As of this revision the size gate is not a single ceiling: `TEXT_FILE_EXTENSIONS`
(`scanner.ts:77#"TEXT_FILE_EXTENSIONS"`, currently just `.md`) gets `maxTextFileSizeBytes` /
`ORACLE_MAX_TEXT_FILE_SIZE` (default 2 MB) instead of the general
`maxFileSizeBytes` / `ORACLE_MAX_FILE_SIZE` (default 500 KB). Rationale: the
ceiling exists to bound how much a single file costs to read fully into
memory before chunking; a large markdown file (a CHANGELOG, a design doc) is
harmless there because the splitter chunks it downstream regardless of size,
so it shouldn't compete with the ceiling sized for arbitrary source files.
Both `scanner.ts` and `watch.ts` compute `isTextType = TEXT_FILE_EXTENSIONS.has(ext)`
and select the applicable limit before the `stat` call. Every `SkippedFile` /
`LoadResult` "too-large" now also carries `limitEnvVar` (`"ORACLE_MAX_FILE_SIZE"`
or `"ORACLE_MAX_TEXT_FILE_SIZE"`, `scanner.ts:125-128#"limitEnvVar?:"`), so WARNING lines name the
knob that actually needs raising instead of always blaming
`ORACLE_MAX_FILE_SIZE` — a markdown file over the *text* ceiling used to (before
the per-type ceiling existed) produce a WARNING telling the operator to raise the wrong env var.

## Config: two size ceilings, same fail-loud parse

`src/config.ts`:
- `maxFileSizeBytes: z.number().int().positive().default(500_000)` (`config.ts:86#"default(500_000)"`)
  and `maxTextFileSizeBytes: z.number().int().positive().default(2_000_000)`
  (`config.ts:98#"default(2_000_000)"`). Both positive int required.
- Env plumbing: `maxFileSizeBytes: parseMaxFileSizeBytes(process.env.ORACLE_MAX_FILE_SIZE)`
  and `maxTextFileSizeBytes: parseMaxTextFileSizeBytes(process.env.ORACLE_MAX_TEXT_FILE_SIZE)`
  (`config.ts:173-174#"parseMaxTextFileSizeBytes(process.env.ORACLE_MAX_TEXT_FILE_SIZE)"`).
- `parseMaxFileSizeBytes(raw)` (`config.ts:205-207#"return Number(raw);"`) and the near-identical
  `parseMaxTextFileSizeBytes(raw)` (`config.ts:215-217#"return Number(raw);"`): `undefined` or
  `raw.trim() === ""` returns `undefined` (schema default applies); otherwise
  `Number(raw)` is handed straight to `configSchema.parse`, where
  `.int().positive()` rejects `NaN`/`0`/negative and `loadConfig` throws. Two
  separate functions (not a shared parser), mirroring the existing
  `parseExtensionsList` / `parseCsvList` pattern of one function per env var.
  (A third, near-identical `parseLlmTimeoutMs(raw)`
  (`config.ts:223-225#"return Number(raw);"`) was added alongside these for
  `ORACLE_LLM_TIMEOUT_MS`; same contract, unrelated env var, out of this doc's
  ingest-size-limit scope.)
- The empty-`.env`-line contract is spelled out in the comment at `config.ts:197-204#"silently resolve to some other limit."`:
  an `ORACLE_MAX_FILE_SIZE=` line "must not crash the CLI" (treated as unset), but
  "A typo'd ORACLE_MAX_FILE_SIZE must fail loudly, not silently resolve to some other limit."
  Same contract for `ORACLE_MAX_TEXT_FILE_SIZE`.

## Scanner: three skip classes, only two reported

`src/ingest/scanner.ts`, `walkRepo` → inner `walk`:
- **Per-type limit selection, then stat before read** (`scanner.ts:201-214#"continue;"`):
  `isTextType = TEXT_FILE_EXTENSIONS.has(ext)` picks `limitBytes` (`maxTextFileSizeBytes`
  or `maxFileSizeBytes`); `const st = await stat(fullPath);` then
  `if (st.size > limitBytes)` calls
  `onSkip({... reason: "too-large", sizeBytes, limitBytes, limitEnvVar})`
  and `continue` — the size decision is made in true bytes, before the file is read
  into memory (replacing the old in-memory `content.length > 200_000` char count).
- **Empty files skip SILENTLY** (`scanner.ts:217-221#"if (!content.trim()) continue;"`): after `readFile`, `if (!content.trim()) continue;`
  with no `onSkip`. Comment: "silent skip on purpose — unlike too-large/read-error, it is not an anomaly worth a WARNING line."
- **Read errors are reported, not swallowed** (`scanner.ts:231-240#"read-error"`): the `catch (err)`
  calls `onSkip({... reason: "read-error", message})`. Non-throwing (one bad file
  must not kill the scan) but never a bare empty catch. Read errors have no
  `limitEnvVar` (only "too-large" does).
- `SkippedFile.reason` is `"too-large" | "read-error"` (`scanner.ts:122#"reason:"`); the empty
  case has no reason value because it is never surfaced.
- `onSkip` defaults to a no-op (`scanner.ts:155#"const onSkip"`); the indexer and watch pass a real one.
- **Asymmetry:** three skip classes (too-large, read-error, empty); only two are
  ever reported.

## Runner: skip count flows into the summary, the WARNING lines, and repo_skip_meta

`src/ingest/runner.ts`:
- `IndexSummary` carries `filesSkipped: number` and `skippedFiles: Array<{repo, relativePath, reason, sizeBytes?, limitBytes?, limitEnvVar?, message?}>` (`runner.ts:41-49#"message?: string;"`).
- `walkOptions.onSkip` pushes each skip into `skippedFiles` (`runner.ts:118-119#"skippedFiles.push({"`).
- **Loud per-file + total reporting** (`runner.ts:196-208#"to index larger files."`): for each skip, `warn(...)`
  emits `WARNING: skipped <path> — <bytes> bytes > <limitEnvVar>=<limit>`
  (too-large; the env var name is per-skip, not hardcoded) or `WARNING: skipped <path> — read error: <message>`, then a one-line
  total naming both env vars as candidates to raise.
- The `warn` sink defaults to a no-op (`runner.ts:79#"const warn = options.warn"`); the CLI routes it to stderr,
  MCP and zero-skip runs stay quiet.
- `formatIndexSummary` folds the count into the **files** segment (`runner.ts:388-395#"skipped)."`):
  `... ${summary.filesNew} new, ${summary.filesPruned} pruned, ${summary.filesSkipped} skipped).`
- **As of this revision**, after the WARNING loop, `runIndex` also tallies skips per
  repo (`sizeCount`, `errorCount`, up to `SKIP_EXAMPLES_LIMIT = 5` example
  paths; `runner.ts:32#"SKIP_EXAMPLES_LIMIT = 5"`, `runner.ts:219-228#"store.setRepoSkipSummary("`) and calls `store.setRepoSkipSummary(repo.name, ...)`
  for **every** discovered repo, including a flat-0 summary for repos that
  skipped nothing this run. That flat-0 overwrite (not a conditional "only
  write when > 0") is what makes the persisted count reflect the LAST run
  rather than accumulate; see sqlite-store.ts's `repo_skip_meta` table and
  `oracle_list_repos` / CLI `list-repos`, which surface it via
  `format-freshness.ts`'s `formatRepoLine` whenever a repo's skipped total is
  non-zero, broken down by reason with up to five example paths. A repo whose
  files were skipped in their entirety (no docs at all) is now surfaced too
  by `listRepos`' widened query, not just repos with a mix of indexed and
  skipped files (sqlite-store.ts's `listRepos` compiled statement). A second
  sweep, `pruneOrphanRepoSkipMeta(discoveredRepos)` (`runner.ts:94-99#"store.pruneOrphanRepoSkipMeta("`),
  removes `repo_skip_meta` rows for repos no longer discovered on disk at
  all; it deliberately does NOT key off `docs` the way `pruneOrphanRepoMeta`
  does, since an all-skipped repo has zero docs while still being live.

## MCP: only the count reaches oracle_reindex, not the per-file lines

`src/mcp-server.ts` oracle_reindex handler (`mcp-server.ts:260-263#"formatIndexSummary(summary)"`):
`const summary = await runIndex(cfg);` then returns `formatIndexSummary(summary)`.
- `runIndex` is called **without** a `warn` sink, so the per-file `WARNING:` lines
  are suppressed on the MCP path (they only appear on CLI stderr).
- The **skip count** still reaches the MCP reindex summary via
  `formatIndexSummary`'s files segment. So: count → MCP; per-file lines → CLI only.
- The persisted per-repo `repo_skip_meta` tally (see above) is independent of
  this and reaches MCP either way: `oracle_list_repos` reads it straight from
  the store, not from a `runIndex` return value.

## The load-bearing duplication (watch.ts is a separate enforcement site)

`src/watch.ts` does **not** call `scanner.ts`. `loadScannedFile` (`watch.ts:150-185#"if (!content.trim()) return"`)
independently reimplements the stat-first gate, including the per-type ceiling:
- `const limit = maxFileSizeBytes ?? DEFAULT_MAX_FILE_SIZE_BYTES;` and
  `const textLimit = maxTextFileSizeBytes ?? DEFAULT_MAX_TEXT_FILE_SIZE_BYTES;`,
  then `isTextType = TEXT_FILE_EXTENSIONS.has(ext)` picks `effectiveLimit`;
  `const st = await stat(absolutePath); if (st.size > effectiveLimit) return {kind:"too-large", ..., limitEnvVar}` (`watch.ts:163-181#"limitEnvVar: isTextType"`).
- Empty check `if (!content.trim()) return {kind:"empty"}` (`watch.ts:185#"if (!content.trim()) return"`).
- It is fed both config values: the caller passes `config.maxFileSizeBytes` AND
  `config.maxTextFileSizeBytes` into `loadScannedFile(...)` (`watch.ts:309-314#"config.maxTextFileSizeBytes,"`).

**watch reports too-large loudly too — verified.** In `flush`, `loaded.kind === "too-large"`
emits `console.warn("WARNING: skipped <path> — <bytes> bytes > <limitEnvVar>=<limit>")`
(`watch.ts:324-329#"store.deleteByFile("`) and unindexes any stale vectors for that file. `loaded.kind === "empty"`
is a **silent** skip (`watch.ts:336-340#"store.deleteByFile("`, mirroring scanner.ts), though it still clears
stale vectors. Note watch reports via `console.warn` directly, whereas runner routes
through an injected `warn` sink — different mechanisms, same "loud" outcome. Watch
does **not** write to `repo_skip_meta` — that persistence is `runIndex`-only (a
full run), so a repo touched only by `npm run watch` for a long stretch keeps
whatever `repo_skip_meta` row its last full `npm run index` / `oracle_reindex`
left behind (console warnings still fire live either way).

**Consequence:** the size/empty gate lives in two places (`scanner.walkRepo`
`scanner.ts:201-221#"if (!content.trim()) continue;"` and `watch.loadScannedFile`
`watch.ts:150-185#"if (!content.trim()) return"`). A fix or
behaviour change to one does **not** automatically apply to the other. Any change
to the threshold semantics, the per-type extension set, the empty-file rule, or
the reporting shape must be made in **both** files, or full-index and
watch-mode behaviour will silently diverge. `TEXT_FILE_EXTENSIONS` itself is
defined once in `scanner.ts` (`scanner.ts:77#"TEXT_FILE_EXTENSIONS"`) and imported by `watch.ts`, so
the *set* of text extensions can't drift even though the ceiling-selection
logic is duplicated.

The `.codebase-oracle-skip` sentinel, unlike the size gate, is shared rather
than duplicated: both sides use the one `SKIP_SENTINEL` constant. The scanner
prunes any directory that contains it (`scanner.ts:169-170#"return;"`), and
watch's `enqueueFileEvent` drops an add/change event whose directory, or any
directory above it up to the repo root, contains it, through the scanner's
`isUnderSkipSentinel` helper (`scanner.ts:27#"export function isUnderSkipSentinel("`,
`watch.ts:390#"if (isUnderSkipSentinel(absolutePath, repoRoot)) return;"`).
The check runs again at flush time (`watch.ts:303#"if (isUnderSkipSentinel(ev.absolutePath, evRepoRoot)) continue;"`),
so an edit queued before a sentinel appeared, inside one debounce window, is
not embedded. A sentinel directory is therefore left out by `npm run watch` as
well as `npm run index`. Watch only filters events: vectors already indexed from
a directory that later gains a sentinel stay until the next full index run
prunes them, and removing a sentinel does not back-fill (watch has no
subtree-scan path and mirroring the scan's skip-dir, extension and size rules
there would be a third copy of them), so files already in the formerly skipped
subtree are picked up by the next full index run or their next change. The
helper resolves both arguments, so a `./`-prefixed path or a root with a
trailing slash is compared like with like.

## Why the reporting is loud (historical)

Per `CHANGELOG.md` `[0.10.0]` (2026-07-04): `agent-tasks/backend/src/routes/tasks.ts`
(207,716 bytes) "was silently dropped by the old `content.length > 200_000` check;
it and any file like it are now reported, not swallowed" (`CHANGELOG.md:#0.10.0`). The same
release moved the limit to a `stat`-first byte check "in both the scanner and
`watch.ts`, which previously duplicated the old limit as its own `MAX_FILE_BYTES`
constant" (`CHANGELOG.md:#0.10.0`) — i.e. the duplication predates the fix and was carried
forward, not introduced by it. The per-type ceiling (`[0.11.0]` in
CHANGELOG.md as of this revision) is the direct descendant of that
same "don't silently drop large files" invariant, applied to the specific case
of a large markdown file that the general ceiling had no reason to treat like
an oversized generated JS bundle.
