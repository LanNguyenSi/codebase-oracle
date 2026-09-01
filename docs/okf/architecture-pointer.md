---
type: overview
title: Architecture — where the pipeline is documented
description: Pointer doc — ../architecture.md is the authoritative pipeline overview (ingest, chunking, OKF frontmatter metadata, embeddings, store, incremental indexing, watch); this entry only names what it does NOT cover.
tags: [overview, architecture, pointer]
timestamp: 2026-09-01T06:58:38Z
sources:
  - docs/architecture.md
---

# Architecture — pointer

[../architecture.md](../architecture.md) is authoritative and current for the
ingest pipeline, chunking, the four `fm*` frontmatter metadata keys, embeddings,
the SQLite + sqlite-vec store, incremental indexing, and watch mode. It is not
duplicated here.

What it does **not** cover, and where to read instead:

- **Sources-expansion** (`oracle_search`-time injection of chunks for files a
  doc's `fmSources` points at) is absent from it entirely. See
  [sources-expansion.md](sources-expansion.md). Note it is a different mechanism
  from the `Pointers (from OKF sources metadata):` section that
  `architecture.md` does describe for `oracle_query`.
- **The two enforcement sites for the ingest size limit** (scanner and watch
  implement it independently). See
  [ingest-size-limit-enforcement.md](ingest-size-limit-enforcement.md).
