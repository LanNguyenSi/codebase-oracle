# codebase-oracle

RAG-powered codebase Q&A using LangChain.js.

## Quick Start

```bash
# Set API keys
export OPENAI_API_KEY=...      # Required when ORACLE_EMBEDDING_PROVIDER=openai
export ANTHROPIC_API_KEY=...   # Optional, for Claude answers

# Optional: use Ollama as OpenAI-compatible provider
# export ORACLE_EMBEDDING_PROVIDER=ollama
# export ORACLE_LLM_PROVIDER=ollama
# export ORACLE_OLLAMA_BASE_URL=http://localhost:11434/v1
# export OLLAMA_API_KEY=ollama
# export ORACLE_EMBEDDING_MODEL=nomic-embed-text
# export ORACLE_LLM_MODEL=llama3.1

# Index all repos
npm run index

# Ask a question
npm run query -- "how does the authentication middleware work?"

# Use as MCP server in Claude Code
claude mcp add codebase-oracle -- npx tsx src/mcp-server.ts
```

## Architecture

- **Ingest**: scanner.ts walks the scan root, splitter.ts chunks with code-aware boundaries
- **Store**: OpenAI-compatible embeddings (OpenAI/Ollama) + in-memory vector store (persisted to ~/.codebase-oracle/)
- **Retrieval**: LangChain RAG chain with Claude/OpenAI/Ollama for answer generation
- **MCP**: 3 tools — oracle_query, oracle_search, oracle_list_repos

## Commands

- `npm run index` — re-index all repos
- `npm run query -- "<question>"` — ask a question
- `npm run dev -- search "<query>"` — raw vector search
- `npm run mcp` — start MCP server (stdio)
- `npm test` — run tests
