# Log

<!-- Add new entries at the top, newest first. -->

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
  quote ("so embedding and LLM live..." vs config.ts:52-55's actual "lets
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
  (src/index.ts:98, verified against `search --help`).

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
