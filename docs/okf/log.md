# Log

<!-- Add new entries at the top, newest first. -->

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
