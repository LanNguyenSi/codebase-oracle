# codebase-oracle

RAG-powered codebase Q&A for the Pandora ecosystem using LangChain.js.

## Quick Start

```bash
# Set API keys
export OPENAI_API_KEY=...      # Required for embeddings
export ANTHROPIC_API_KEY=...   # Optional, for answer generation (falls back to OpenAI)

# Index all repos
npm run index

# Ask a question
npm run query -- "how does task_finish handle autoMerge?"

# Use as MCP server in Claude Code
claude mcp add codebase-oracle -- npx tsx src/mcp-server.ts
```

## Architecture

- **Ingest**: scanner.ts walks ~/git/pandora, splitter.ts chunks with code-aware boundaries
- **Store**: OpenAI embeddings + MemoryVectorStore (persisted to ~/.codebase-oracle/index.json)
- **Retrieval**: LangChain RAG chain with Claude/OpenAI for answer generation
- **MCP**: 3 tools — oracle_query, oracle_search, oracle_list_repos

## Commands

- `npm run index` — re-index all repos
- `npm run query -- "<question>"` — ask a question
- `npm run dev -- search "<query>"` — raw vector search
- `npm run mcp` — start MCP server (stdio)
