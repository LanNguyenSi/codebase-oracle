# Architecture

How codebase-oracle turns a directory of git repos into a queryable semantic index.

## Pipeline

```
$ORACLE_SCAN_ROOT/**/*.ts,md,prisma
        |
        v
   [Scanner]  ── walk repos, filter files
        |
        v
   [Splitter] ── code-aware chunking (function/class boundaries)
        |
        v
  [Embeddings] ── OpenAI text-embedding-3-small
        |
        v
 [Vector Store] ── sqlite-vec (SQLite virtual table, cosine distance, WAL)
        |
        v
  [RAG Chain]  ── retrieved chunks + question → Claude/OpenAI/Ollama → cited answer
```

The index pipeline scans all git repos under a root directory, splits source files into chunks with language-aware boundaries, embeds them via OpenAI-compatible APIs (OpenAI or Ollama), and stores the vectors locally. Queries retrieve the most relevant chunks and feed them to an LLM for answer generation with source citations.

## Chunking

The splitter uses LangChain's code-aware splitters where available, falling back to a recursive character splitter for text. Chunks record `lineStart` / `lineEnd` so search results can render as `path:line_start-line_end (repo)`. Older chunks indexed before the line-number rollout fall back to the bare `filePath` form.

Markdown files that lead with a YAML frontmatter block (`---` ... `---` as the first lines) get four additional flat metadata keys: `fmType`, `fmTitle`, `fmTags`, and `fmSources`, read from the frontmatter's `type`, `title`, `tags`, and `sources` fields respectively. Each key is included only when the source field is present with the right type (non-empty string for `fmType`/`fmTitle`, an array of non-empty strings for `fmTags`/`fmSources`); anything else is silently omitted, and the frontmatter text itself is left in the chunked content rather than stripped out. A malformed frontmatter block fails soft: the ingest logs a warning naming the file and continues without any `fm*` keys, never throwing.

## OKF-aware retrieval

`oracle_search` (CLI, stdio MCP, and HTTP MCP) accepts optional `type` and `tags` filters over `fmType` / `fmTags`: `type` is a strict-equality match, `tags` requires every listed tag to be present (contains-all). Both only match chunks that HAVE the corresponding field; chunks without frontmatter metadata are excluded whenever a filter is set. Matching results show `fmType` as a `[type]` tag in the header and, when present, a `sources: ...` line built from `fmSources`. `oracle_query` needs no extra params: retrieved chunks' `fmSources` are mechanically unioned (deduped, rank-ordered, capped at 10) into a `Pointers (from OKF sources metadata):` section appended after the answer's sources list, or omitted when no retrieved chunk carries `fmSources`. See [docs/mcp.md](mcp.md#okf-frontmatter-filters-type--tags) for details and examples.

## Embeddings

By default, OpenAI's `text-embedding-3-small` (1536 dimensions). Override with `ORACLE_EMBEDDING_PROVIDER=ollama` and a local embedding model like `nomic-embed-text` to keep everything off the network. See [docs/configuration.md](configuration.md) for the full env var list.

## Vector store: sqlite-vec

Since v0.3.0 the store is a single SQLite file (`store.db`) opened in WAL mode, backed by [sqlite-vec](https://github.com/asg017/sqlite-vec). Implications:

- Writes and reads can safely happen from different processes on the same store.
- A running stdio or HTTP MCP server sees `npm run watch` writes on its next query without restarting.
- Cosine distance for similarity, computed by the virtual table.

## Embedding fingerprint

The store keeps a fingerprint (`embeddingProvider`, `embeddingModel`, `dimension`) in the `meta` table. On load, codebase-oracle refuses to run against a different provider/model: you would get silent garbage otherwise, because the query vector and the stored vectors would live in different embedding spaces (or differ in dimension, producing `NaN` scores).

If you change `ORACLE_EMBEDDING_PROVIDER` or `ORACLE_EMBEDDING_MODEL`, the next `npm run index` / `npm run query` / MCP call will fail fast with a clear message telling you to either delete `~/.codebase-oracle/store.db` (to re-embed with the new model) or revert the env change. There is no automatic migration: the choice is yours.

## Incremental indexing

Indexing is incremental when `ORACLE_VECTOR_STORE=directory`: unchanged files are reused from persisted vectors (via file hashes), and only new/changed files are re-embedded. Progress is checkpointed batch-by-batch during embedding, so interrupted runs can resume without redoing all completed batches.

## Watch mode

`npm run watch` runs a [chokidar](https://github.com/paulmillr/chokidar) watcher over the scan root. File add/change/delete events are accumulated and, after a quiet period (default 3 s), processed in one batch: changed files are re-embedded, deleted files drop their vectors, vanished repo roots purge all their vectors. Editor save-storms (e.g. VS Code's atomic-rename trick) collapse into a single re-embed thanks to chokidar's `awaitWriteFinish` plus the debounce. Newly dropped `.git` roots are detected and logged; back-fill them with one explicit `npm run index` so the first-time ingestion is consistent, then watch picks up subsequent edits.

Watch mode is additive: it does not replace `npm run index`, which remains the ground-truth bootstrap path.

## Tech stack

- **LangChain.js**: document splitting, embeddings orchestration, RAG chain
- **OpenAI-compatible APIs**: OpenAI and Ollama for embeddings and LLM
- **Claude** (Anthropic): answer generation with source citations
- **MCP SDK**: Model Context Protocol server for Claude Code integration
- **TypeScript** + **Zod**: type-safe configuration and validation

## Credits

The core idea, exposing a semantic index of docs and code to agents through a single MCP endpoint, was inspired by [andrepester/rag-search-mcp](https://github.com/andrepester/rag-search-mcp). codebase-oracle is a Node/LangChain-flavoured take on the same concept, tuned for multi-repo JavaScript/TypeScript workspaces.
