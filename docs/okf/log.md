# Log

<!-- Add new entries at the top, newest first. -->

- 2026-09-26T04:44:36Z, docs refresh: the `expand_sources` flag-table
  pointer moved from README.md to `docs/cli-reference.md` (the flag table
  itself moved there in a prior README restructure); re-verified and
  re-pointed in `sources-expansion.md` and `index.md`, and re-verified the
  `ORACLE_LLM_TIMEOUT_MS` default-reasoning pointer in
  `provider-enums-and-token-budget.md` against `docs/configuration.md`
  (claim still holds). All three docs re-stamped.

- 2026-09-23T11:37:34Z, release 0.13.0: `package.json`'s version field changed
  in place (line 3, no cited line moved); `index-freshness-vs-code-freshness.md`
  re-checked and re-stamped.

- 2026-09-23T10:28:00Z, task 844aac2c: added an invoke-level overall
  deadline (`chain.invoke(..., { timeout: config.llmTimeoutMs })`) in
  `queryCodebase` alongside the constructor-level timeout, raised the
  `ORACLE_LLM_TIMEOUT_MS` default to 120000 with a `.max(2147483647)` ceiling
  in `src/config.ts`, and corrected the retry-layer comments in
  `src/retrieval/chain.ts`. Citations into `chain.ts` and `config.ts` were
  re-verified and re-pointed in `provider-enums-and-token-budget.md`,
  `sources-expansion.md`, `ingest-size-limit-enforcement.md` and
  `index-freshness-vs-code-freshness.md`; `configuration-pointer.md` was
  re-stamped for its `docs/configuration.md` source (as was
  `provider-enums-and-token-budget.md`, whose claim about that file still holds).

- 2026-09-23T10:00:30Z, task 844aac2c: every LLM constructor in
  `src/retrieval/chain.ts` sets a request timeout and `maxRetries: 0` from a
  new `config.llmTimeoutMs` field (`src/config.ts`, env
  `ORACLE_LLM_TIMEOUT_MS`, documented in `docs/configuration.md`). Shifted
  citations into both files were re-verified and re-pointed in the bundle
  docs that list them, which were re-stamped.
  `provider-enums-and-token-budget.md` gained a bullet naming the timeout knob
  as outside its scope; `ingest-size-limit-enforcement.md` gained a
  parenthetical on the sibling `parseLlmTimeoutMs` parser.

- 2026-09-23T09:41:06Z, task 53112c27: fixed `--help`/`-h`/`--version` exiting 1
  with an `ok:false` JSON document in `--json` mode (`src/index.ts`, catch
  handler inserted after line 308). The
  insertion sits after every citation in `index-freshness-vs-code-freshness.md`
  (the one doc listing `src/index.ts` under `sources:`), so no citation moved;
  its `loadConfig()` call-site lines (74, 90, 150, 192, 221, 247, 263) and the
  `index` command range (70-79) were re-verified against the edited file and
  still name the same content. That doc's timestamp was re-stamped
  (2026-09-23T09:40:18Z, again at 09:50:18Z after a comment-only follow-up
  edit) in a separate commit after the source commit.
  `npx okf-kit@0.10.0 check docs/okf --json` on this commit: 0 errors, 0
  warnings, 0 notices.

- 2026-09-23T06:21:49Z, task df880cbc: added an additive `ok: true` to the `search`,
  `list-repos` and `query` `--json` success documents and a
  `degraded`/`degradedReason` marker on `query`'s LLM-failure raw-context
  fallback (`src/retrieval/chain.ts`, `src/format-json.ts`). The 9-line
  insertion in `chain.ts` shifted later lines by 7 or 9; every `chain.ts`
  citation in `provider-enums-and-token-budget.md` and
  `sources-expansion.md` (the two docs listing it under `sources:`) was
  re-verified and re-pointed, and both docs were re-stamped. Seven bare
  line-number citations in `sources-expansion.md`'s "Things that silently
  break injection" bullet already pointed at the wrong lines at base
  9b2853c (for example the old 588 landed on a comment, not the
  `parentRepo.length === 0` check); they now name their real targets.
  Base 9b2853c checked 0 errors, 0 warnings, 0 notices. This change
  introduced the warnings it then fixed: the `chain.ts` shift broke the two
  docs' citations, and the new `CHANGELOG.md` `[Unreleased]` entry moved
  the 0.10.0 section down by seven lines, so bare CHANGELOG line-number
  mentions in older entries below landed on blank lines. Those older
  entries changed only in citation tokens, each keeping the target it
  meant when written: line-101 mentions now name the "was silently
  dropped" quote line and line-97 mentions the MAX_FILE_BYTES-quote line;
  the 2026-09-21 entry spells out its line-97 and line-101 mentions in
  words (needed: the literal base wording re-triggers a blank-start-line
  warning) and qualifies its sqlite-store continuation citation with the
  file name (kept for robustness; the bundle also checks clean without
  it). `npx okf-kit@0.10.0 check docs/okf --json` on this commit: 0 errors,
  0 warnings, 0 notices.

- 2026-09-22T06:05:17Z, task e53a6d9d (implementer): added a `## [Unreleased]`
  entry to `CHANGELOG.md` (release guards) for a codebase-oracle-local
  task, no release cut. The ten added lines shifted `CHANGELOG.md` by ten
  lines, same recurring effect the 2026-09-21 entry below already
  described: this log's own historical bare CHANGELOG-line-number mentions
  below (none of them a live doc citation; the live docs cite
  `## [0.10.0]` by heading) again landed on now-blank lines, `okf-kit
  check` reporting 11 `citations-resolve` `blank-start-line` warnings.
  None of the touched files (`release.yml`, `README.md`, `CHANGELOG.md`)
  are a declared `sources:` entry of any doc in this bundle, so no doc
  needed re-verification against them; this is only the same pre-existing
  bare-line-number fragility in this log's own prose. Re-pointed each of
  those old-87 mentions ten lines forward to land on line ninety-seven,
  and each old-91 mention ten lines forward to land on line one hundred
  one (the same ten-line shift, restoring the exact non-blank lines those
  numbers already landed on before this cut, written out in words here so
  this entry itself does not add a new resolvable file:line citation); no
  other narrative text changed. `npx okf-kit@0.14.0 check docs/okf --json`
  after this fix: 0 errors, 0 warnings, 0 notices.

- 2026-09-21T11:16:00Z, 0.12.0 release cut (orchestrator): `package.json` changed twice
  since `index-freshness-vs-code-freshness.md` was stamped (the vitest pins in
  the 2026-09-11 CVE sweep, then the version field in this cut), both as
  same-line replacements. Re-verified every `package.json` line the doc cites
  (the `bin` block, `build`, `dev`, `index`, `mcp`, `serve`, `prepublishOnly`)
  against the file: unchanged in place; the doc's version-flow passage
  (`package.json` through `src/version.ts` into the MCP handshake) is what the
  0.12.0 dogfood observed. Re-stamped. `sources-expansion.md` cited the 0.10.2
  release as a bare `CHANGELOG.md` line number that had drifted into the
  unreleased section; it now cites the section by heading
  (`` `CHANGELOG.md:#0.10.2` ``), the form this bundle's index prescribes, so
  later cuts cannot shift it; its `getFirstChunkByFileInternal` range in
  `src/store/sqlite-store.ts` was five lines early and now reads
  `src/store/sqlite-store.ts:777-789`.
  Re-stamped. The cut grew `CHANGELOG.md` by seven lines above every released
  section (the heading, its blank line, five lines of added Security
  bullets). The bare `CHANGELOG.md` line-97 and line-101 mentions in
  this log's older entries are history, not a live citation: each release
  cut shifts them forward again, and the newest entry at the top of this
  log states the shift that is current as of that cut (see the
  2026-09-22T06:05:17Z entry above for the shift current as of this
  writing). The check stops reporting them as blank-start-line warnings
  only when the lines they land on happen to be non-blank. The live docs
  cite those passages by heading (`` `CHANGELOG.md:#0.10.0` ``) and are
  unaffected.

- 2026-09-08T04:38:41Z, task 8cfca118 (implementer): applied the pattern
  harness adopted (PRs #514, #516) to close the recurring `CHANGELOG.md`
  `sources-fresh` re-stale for `ingest-size-limit-enforcement.md`: any
  CHANGELOG edit anywhere in the repo used to re-stale this doc, and an
  unrelated author could not honestly re-stamp it. Re-verified both
  passages quoting the `[0.10.0]` entry ("was silently dropped by the old
  `content.length > 200_000` check; it and any file like it are now
  reported, not swallowed" and "in both the scanner and `watch.ts`, which
  previously duplicated the old limit as its own `MAX_FILE_BYTES`
  constant") against `CHANGELOG.md`'s `## [0.10.0] - 2026-07-04` section:
  both quotes match verbatim inside it. Re-pointed both citations from
  line numbers (the "was silently dropped" quote line, and the
  MAX_FILE_BYTES-quote line) to the heading form
  `` `CHANGELOG.md:#0.10.0` ``, and dropped `CHANGELOG.md` from the doc's
  frontmatter `sources:`. Doc re-stamped (`timestamp: 2026-09-08T04:38:41Z`).
  Also added the maintenance rule to `docs/okf/index.md` ("do not list
  `CHANGELOG.md` under sources; cite release sections by heading"), one
  sentence, matching harness's own index.md rule.
  Negative control (planned: append a throwaway line to `CHANGELOG.md`
  uncommitted, run `npx okf-kit@0.10.0 check --json docs/okf`, expect 0
  findings and no `sources-fresh` finding naming `CHANGELOG.md`, then
  revert). Observed: ran before this commit, on the working tree with
  both edits already applied but not yet committed: 0 errors/warnings/
  notices across the whole bundle; no `sources-fresh` finding for
  `ingest-size-limit-enforcement.md` or any other doc. Reverted with
  `git checkout -- CHANGELOG.md`.

- 2026-09-07T11:02:16Z, sources-fresh + blank-start-line fix (task ee959161):
  after the okf-kit 0.10.0 fleet bump (PR #99), `okf-kit check --json
  docs/okf` flagged index-freshness-vs-code-freshness.md and
  ingest-size-limit-enforcement.md STALE (their `src/index.ts` and
  `CHANGELOG.md` sources both changed 2026-09-04) plus four
  `citations-resolve` `blank-start-line` warnings. Re-verified every
  citation in both docs against HEAD.
  index-freshness-vs-code-freshness.md: 24 checked, 2 corrected (the
  `loadConfig()` call-site list, old line numbers 55, 69, 118, 149, 172,
  193 and 209 in `src/index.ts`, re-pointed to `src/index.ts:74, 90, 150,
  192, 221, 247, 263`; the index-command range, old lines 51 to 56 in
  `src/index.ts`, re-pointed to `src/index.ts:70-79`; both shifted by a
  `-p, --path <path>` option and other commands added ahead of them since
  the doc was last verified).
  ingest-size-limit-enforcement.md: 32 checked, 2 corrected (old line 82 in
  `CHANGELOG.md`, re-pointed to the "was silently dropped" quote line, for
  the "was silently dropped" 0.10.0 quote; old line 78 in `CHANGELOG.md`,
  re-pointed to the MAX_FILE_BYTES-quote line, for the "MAX_FILE_BYTES
  constant" quote; both shifted by the `[Unreleased]` CLI `--json`-flag
  entry added to CHANGELOG.md).
  That second correction is also one of the four blank-start-line fixes:
  the MAX_FILE_BYTES-quote line starts on content. The other three
  blank-start-line fixes are inside this log's own earlier historical
  entries, and only the cited range was re-pointed there, the surrounding
  narrative is left as written at the time: the 2026-09-01T07:30:00Z
  entry's old line 80 in `CHANGELOG.md` is now the "was silently dropped"
  quote line and its old line 76 in `CHANGELOG.md` is now the
  MAX_FILE_BYTES-quote line (the same two CHANGELOG.md quotes above,
  shifted further since that entry was written), and the
  2026-08-22T05:21:41Z entry's old line 98 in `src/index.ts` is now
  `src/index.ts:132` (the `-g, --path-glob <glob>` commander option
  definition, text unchanged, line shifted).
  All other declared sources for both docs (`package.json`,
  `src/store/sqlite-store.ts`, `src/mcp-server.ts`, `src/version.ts`,
  `docs/architecture.md`, `src/config.ts`, `src/ingest/scanner.ts`,
  `src/ingest/runner.ts`, `src/watch.ts`) last changed before both docs'
  prior timestamp (`package.json` a few minutes after it, in the same
  release commit that produced that stamp, so no drift), so their
  existing citations were re-checked and left
  as is. No claim or content changed in either doc. Noted for a follow-up,
  out of scope here: ingest-size-limit-enforcement.md's only remaining
  dependence on `CHANGELOG.md` is those two historical 0.10.0 quotes;
  citing the release section by heading and dropping `CHANGELOG.md` from
  `sources:` (the pattern harness's index.md maintenance rule already uses)
  would stop future unrelated CHANGELOG.md edits from re-staling this doc.
  Re-stamped both docs. `okf-kit check --json docs/okf` after this commit:
  0 errors, 0 warnings, 0 notices.

- 2026-09-02T04:47:51Z, okf-kit pin bump (task 44ee799a, fleet parity):
  bumped .github/workflows/okf-staleness.yml's pin from okf-kit@0.6.0 to
  okf-kit@0.9.0 to match the other OKF bundle repos (measured: 0.8.0 and 0.9.0 report identical findings for this
  bundle, confirmed again here). Before this change: `okf-kit check --json
  docs/okf` reported 0 errors, 0 warnings, 5 notices (same count under
  both 0.8.0 and 0.9.0). All 5 notices are `citations-resolve` /
  `unresolved-ambiguous` hits on bare `runner.ts:NN` and `config.ts:NN`
  citations inside this log's own historical entries (2026-09-01T07:25:32Z
  and 2026-09-01T07:43:40Z entries above), ambiguous because the repo also
  has tests/eval/runner.ts and tests/eval/corpus/config-toy/src/config.ts.
  Re-opened each cited span in src/ingest/runner.ts and src/config.ts,
  confirmed line content is unchanged and still matches what those
  historical entries describe (SKIP_EXAMPLES_LIMIT, IndexSummary, the
  `warn` no-op default, the prune-sweep call, and the
  embedding/LLM-provider comment), then re-pointed all five citations to
  the fully-qualified src/ingest/runner.ts and src/config.ts paths so they
  resolve unambiguously. No claim in any doc changed. After this change:
  0 errors, 0 warnings, 0 notices.

- 2026-09-01T07:43:40Z, review-round-2 verification fix (task ee173398): the style
  commit that removed em dashes from round-2 prose touched
  `src/store/sqlite-store.ts` and `docs/configuration.md` after the
  previous re-stamp, so `okf-kit check` flagged configuration-pointer.md,
  provider-enums-and-token-budget.md, index-freshness-vs-code-freshness.md
  and sources-expansion.md STALE again. The touched lines were punctuation
  only; re-verified that no cited span or claim moved, re-stamped all four.
  Also added a rule-level comment on `pruneOrphanRepoSkipMeta` naming the
  SQLite bound-parameter ceiling its NOT IN clause would hit at roughly a
  thousand discovered repos (reviewer low finding), which does not change
  any claim these docs carry. Verdict after this commit: 0 errors,
  0 warnings, the 5 pre-existing notices.

- 2026-09-01T07:30:00Z, review-round fix (task ee173398, round 2, follow-up):
  `okf-kit check` after the round-2 commit flagged three more docs STALE:
  architecture-pointer.md, configuration-pointer.md, and
  provider-enums-and-token-budget.md all list docs/architecture.md and/or
  docs/configuration.md in `sources:`, both of which the round-2 commit
  edited (the ORACLE_MAX_TEXT_FILE_SIZE trade-off sentence and the
  watch-mode skip-persistence sentence). Checked all three against the
  current doc content: none of their claims (topic coverage, the two
  provider enums, the token-budget asymmetry) were affected by those two
  sentences, so no citation needed re-pointing; restamped only. Separately,
  the same round-2 CHANGELOG.md edit (net +7 lines in `[Unreleased]`) shifted
  two citations in ingest-size-limit-enforcement.md's historical section that
  the first check run had missed (only sources-fresh and the new prune
  citation were checked by hand there): the "was silently dropped" 0.10.0
  quote's line number moved from 71 to the "was silently dropped" quote
  line, and the "MAX_FILE_BYTES constant" quote's moved from 67 to the
  MAX_FILE_BYTES-quote line. `okf-kit check` (0.8.0)
  now reports 0 errors, 0 warnings; the 5 remaining NOTICEs are pre-existing bare
  `runner.ts:NN` / `config.ts:NN` ambiguous-citation notices in THIS log's
  own historical entries, already flagged this way before round 2 (see the
  first log entry's own note that such citations describe past states, not
  live claims, and are left as is).

- 2026-09-01T07:25:32Z, review-round fix (task ee173398, round 2): sqlite-store.ts
  edits (widened `listRepos`, added `pruneOrphanRepoSkipMeta`) shifted every
  cited line at or below the `SqliteStore` interface. Re-pointed
  index-freshness-vs-code-freshness.md's WAL pragma citations (207-213 →
  218-224, 209-213 → 220-224), listRepos/similaritySearch read citations
  (441,496,519 → 476,531,554), writeEpoch citations (148-150,561,585,749-758
  → 159-161,596,620,810-818), the `last_indexed_at` listRepos citation
  (303-316 → 323-351), and the assertCompatibleWithConfig citation (425-437 →
  460-473). Re-pointed sources-expansion.md's getFirstChunkByFile citations
  (336-338 → 371-373, 711-723 → 772-784). runner.ts edits (new
  pruneOrphanRepoSkipMeta call) shifted ingest-size-limit-enforcement.md's
  onSkip/warning-loop/formatIndexSummary/skip-tally citations (107-117 →
  118-128, 185-199 → 196-210, 377-388 → 388-399, 201-221 → 212-232); added a
  citation for the new prune sweep (src/ingest/runner.ts:94-104). SKIP_EXAMPLES_LIMIT
  (src/ingest/runner.ts:32), the IndexSummary interface (src/ingest/runner.ts:34-55), and the
  `warn` no-op default (src/ingest/runner.ts:79) were unaffected; the new code was
  inserted after them. Restamped.

- 2026-08-22T07:23:49Z, review-round fix (task 84dcedaa): the src/index.ts
  entry-guard fix (`isMainModule` now realpath-resolves `process.argv[1]`
  before comparing to `import.meta.url`, so the CLI works when reached
  through a symlink such as the npm bin shim) shifted every subsequent line
  in the file. index-freshness-vs-code-freshness.md:135's seven `loadConfig()`
  call-site line numbers (`src/index.ts:...`) and :150's `index` command
  citation were re-measured against current source and fixed; no other
  `src/index.ts:NN` citation exists in docs/okf outside this file (checked
  via `grep -rn 'index.ts:[0-9]' docs/`; the two hits in log.md are dated
  changelog entries describing past states, not live claims, so left as is).
  Restamped.

- 2026-08-22T05:39:39Z, docs-audit follow-up review round (task dd7c19f2):
  the prior entry's "everything else matched" / "real drift" claims for
  index-freshness-vs-code-freshness.md and provider-enums-and-token-budget.md
  were incomplete; a review pass found two more issues the first pass missed.
  index-freshness-vs-code-freshness.md:150 cited `package.json:22` for the
  `index` npm script; line 22 is `dev`, the script is at line 23 (the doc
  already cited :23 correctly elsewhere, at line 73), fixed, restamped.
  provider-enums-and-token-budget.md:33 presented a paraphrase as a verbatim
  quote ("so embedding and LLM live..." vs src/config.ts:52-55's actual "lets
  embedding and LLM live...") was reworded so the quoted span matches the
  source exactly, restamped.

- 2026-08-22T05:21:41Z, docs-audit follow-up (task dd7c19f2): re-verified
  provider-enums-and-token-budget.md against src/config.ts and
  src/retrieval/chain.ts line by line; all cited lines still match, no
  content change, restamped (it went STALE only because its declared
  source docs/configuration.md changed in PR #83). index-freshness-vs-code-freshness.md
  re-verified against package.json/src/store/sqlite-store.ts/src/mcp-server.ts/
  src/version.ts/src/config.ts/src/index.ts/docs/architecture.md: found and
  fixed one real drift, the CLI `loadConfig()` call-site list was missing the
  `migrate-store` command (src/index.ts:189), added it; restamped.
  ingest-size-limit-enforcement.md re-verified against src/config.ts,
  src/ingest/scanner.ts, src/ingest/runner.ts, src/mcp-server.ts, src/watch.ts,
  CHANGELOG.md: found and fixed real drift, every watch.ts line reference had
  shifted +15 lines since the doc was last written (loadScannedFile and its
  too-large/empty branches in `flush`), corrected all six citations; restamped.
  README's CLI flag table also gained the missing `-g, --path-glob <glob>` row
  (src/index.ts:132, verified against `search --help`).

- 2026-08-22T04:51:51Z, docs-freshness-audit round-2 fix (task cecad947):
  mcp.md's tool table and both tool-input summaries now state
  `expand_sources` is a boolean defaulting to `true` rather than just
  listing it among the narrowing filters; index.md's MCP bullet now
  names where the `expand_sources` semantics live, matching its
  siblings' "plus ..." shape; mcp-pointer.md re-verified against the
  now-final mcp.md and restamped (its 04:35:37Z timestamp went stale
  again the moment mcp.md was edited further in the prior fix round; no
  content change needed this time).

- 2026-08-22T04:44:50Z, docs-freshness-audit fix round (task cecad947):
  sources-expansion.md and index.md no longer claim the expand_sources
  parameter itself is undocumented (README.md and mcp.md list it; only the
  dedup/hoist/cap semantics are doc-only-here); mcp.md's tool table and
  example-prompts line now list expand_sources for oracle_search;
  configuration-pointer.md re-verified against docs/configuration.md's
  .env-cwd fix and restamped (no content change needed).

- 2026-08-22T04:35:37Z, docs-freshness-audit follow-up (task cecad947):
  mcp-pointer.md and sources-expansion.md both said docs/mcp.md does not
  document expand_sources; that became false once mcp.md's parameter
  enumeration was fixed in the same pass, so both sentences were updated
  to match and timestamps bumped (mcp-pointer.md's bump was superseded
  by the next fix round; see the entry above).

- 2026-07-16T02:36:27Z, re-verification sweep (task f0121f17): 4 stale docs re-checked
  against current sources. Substantive: sources-expansion's dedup section
  rewritten for the 0.10.2 hoist fix (d165ff85, the doc described the
  pre-fix skip behavior and banned the word the code now uses);
  provider-enums' base-URL paragraph updated for the ab6aad16 guard fix;
  stripped leftover first-person verification footers that had been
  saved into three doc bodies past unmatched code fences (they were
  being indexed and retrieved as content).

- 2026-07-16T01:03:30Z, CI now watches staleness: warn-only
  `okf-kit check` on every PR (.github/workflows/okf-staleness.yml,
  canonical pattern from harness#350).
- , initial 7 docs (4 concept, 3 pointer) authored and verified against
  sources at master cb2dce6 (v0.10.1).
