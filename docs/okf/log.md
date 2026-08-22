# Log

<!-- Add new entries at the top, newest first. -->

- 2026-08-22T04:35:37Z, docs-freshness-audit follow-up (task cecad947):
  mcp-pointer.md and sources-expansion.md both said docs/mcp.md does not
  document expand_sources; that became false once mcp.md's parameter
  enumeration was fixed in the same pass, so both sentences were updated
  to match and timestamps bumped.

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
