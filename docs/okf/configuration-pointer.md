---
type: overview
title: Configuration — where the env vars are documented
description: Pointer doc — ../configuration.md is the authoritative env-var reference and already separates the embedding and LLM provider enums; this entry only points at the token-budget asymmetry it does not mention.
tags: [overview, configuration, env, pointer]
timestamp: 2026-09-01T07:30:00Z
sources:
  - docs/configuration.md
---

# Configuration — pointer

[../configuration.md](../configuration.md) is authoritative and exhaustive for
environment variables, including `ORACLE_MAX_FILE_SIZE` semantics and the two
separate provider enums (`ORACLE_EMBEDDING_PROVIDER`: `openai` | `ollama` |
`stub`; `ORACLE_LLM_PROVIDER`: `auto` | `anthropic` | `openai` |
`openai-compatible` | `ollama`). Not duplicated here.

What it does not mention: the answer LLM's **token budget is set only on the
Anthropic lane**. The OpenAI and OpenAI-compatible lanes construct their client
without a `maxTokens` cap, which produces an empty answer rather than an error
when a thinking model spends the budget on its reasoning field. See
[provider-enums-and-token-budget.md](provider-enums-and-token-budget.md).
