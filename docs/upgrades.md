# Upgrades

## v0.6.5 → v0.7.0

v0.7.0 adds chunk metadata extracted from markdown YAML frontmatter: files whose content leads with a `---`/`---` block get `fmType`, `fmTitle`, `fmTags`, and `fmSources` keys (each omitted when the corresponding field is missing or the wrong type) alongside the existing `repo` / `filePath` / `language` / `lineStart` / `lineEnd` metadata. No on-disk migration is required: existing chunks simply lack the `fm*` keys until their file changes or the store is rebuilt. Run `npm run index` after upgrading to backfill changed files, or delete `store.db` for a full rebuild that picks up frontmatter metadata everywhere.

## v0.3.0 → v0.4.0

v0.4.0 adds line-number rendering in `oracle_search` results (`path:line_start-line_end (repo)`), the `oracle_expand` MCP tool for reading windows around a chunk, and `lastIndexedAt` timestamps in `oracle_list_repos`. No on-disk migration is required: existing v0.3 stores keep working, and chunks indexed before the line-number rollout fall back to the bare `filePath` form. Re-index to pick up line numbers everywhere.

## v0.2.0 → v0.3.0

v0.3.0 moved the on-disk format from `embeddings.jsonl` to a SQLite file (`store.db`) backed by [sqlite-vec](https://github.com/asg017/sqlite-vec). Two upgrade paths:

```bash
# Option A — convert the existing JSONL in place (preserves the index):
npm run migrate-store

# Option B — fresh re-index:
rm ~/.codebase-oracle/embeddings.jsonl
npm run index
```

`migrate-store` reads the JSONL, writes the SQLite store with the same embedding fingerprint, and renames the JSONL to `.embeddings.jsonl.bak` on success. It refuses to run if a `store.db` already exists with data.
