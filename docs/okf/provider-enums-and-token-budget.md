---
type: invariant
title: Two provider enums, and the token budget only one of them sets
description: embeddingProvider and llmProvider are independent enums with independent env vars; only the Anthropic LLM lane caps maxTokens, so an uncapped OpenAI-compatible thinking model can return empty content.
tags: [config, providers, llm, embeddings, gotcha]
timestamp: 2026-09-01T06:58:38Z
sources:
  - src/config.ts
  - src/retrieval/chain.ts
  - docs/configuration.md
---

## Invariant

Provider selection in codebase-oracle is split across **two separate, independently-set enums**. They are routinely conflated because both are "the provider," but embedding and answering are different lanes with different env vars, different default models, and different runtime construction. One of those lanes (Anthropic) caps the response token budget; the other two (OpenAI, OpenAI-compatible) do not — and that asymmetry is a live operational failure mode on this repo's current config.

## The two enums (`src/config.ts:25-32`)

```ts
embeddingProvider: z.enum(["openai", "ollama", "stub"]).default("openai"),
llmProvider: z.enum([
  "auto",
  "anthropic",
  "openai",
  "openai-compatible",
  "ollama",
]).default("auto"),
```

- `embeddingProvider` — env `ORACLE_EMBEDDING_PROVIDER` (`config.ts:110-112`). Values: `openai | ollama | stub`. (`stub` = deterministic hash vectors, integration tests only — `config.ts:22-24`.)
- `llmProvider` — env `ORACLE_LLM_PROVIDER` (`config.ts:113-115`). Values: `auto | anthropic | openai | openai-compatible | ollama`.

They are **independent**: you can embed with `openai` and answer with `ollama` (or any other combination). The comment at `config.ts:52-55` says these keys are kept separate because doing so "lets embedding and LLM live on different providers without leaking keys across lanes."

## Default models (`loadConfig`, `config.ts:117-124`)

```ts
const defaultEmbeddingModel = embeddingProvider === "ollama"
  ? "nomic-embed-text"
  : "text-embedding-3-small";
const defaultLlmModel = llmProvider === "openai"
  ? "gpt-4o-mini"
  : llmProvider === "ollama"
    ? "llama3.1"
    : "claude-sonnet-4-6";
```

- Embedding default: `nomic-embed-text` for `ollama`, else `text-embedding-3-small`. Override: `ORACLE_EMBEDDING_MODEL`.
- LLM default: `gpt-4o-mini` for `openai`, `llama3.1` for `ollama`, else (`auto`, `anthropic`, **and `openai-compatible`**) `claude-sonnet-4-6`. Override: `ORACLE_LLM_MODEL`. Note that `openai-compatible` inherits the Anthropic-named default, so a real endpoint almost always needs `ORACLE_LLM_MODEL` set explicitly.

### What `auto` resolves to (`createLlm`, `src/retrieval/chain.ts:419-462`)

`auto` is not routed by a dedicated branch. It falls through the explicit-provider `if`s and lands on the credential-sniffing tail: `anthropicApiKey` present -> Anthropic (`453-454`); else `openaiApiKey` present -> OpenAI with the hardcoded `OPENAI_AUTO_FALLBACK_MODEL = "gpt-4o-mini"` (`chain.ts:24`, `457-458`); else returns `null` (`461`). `auto` **never** resolves to `openai-compatible` or `ollama` — those require an explicit `ORACLE_LLM_PROVIDER`.

## The gotcha: only the Anthropic lane caps `maxTokens`

Three LLM constructors, three different budget behaviors (`src/retrieval/chain.ts`):

```ts
function createAnthropicLlm(config: Config) {          // 349-356
  return new ChatAnthropic({ ..., temperature: 0, maxTokens: 4096 });
}

function createOpenAILlm(config: Config, modelName: string) {   // 358-367
  return new ChatOpenAI({ ..., temperature: 0, /* NO maxTokens */ });
}

function createOpenAICompatibleLlm(config: Config, isLegacyOllama: boolean) {  // 374-398
  return new ChatOpenAI({ ..., temperature: 0, /* NO maxTokens */ });
}
```

`createAnthropicLlm` sets `maxTokens: 4096` (`chain.ts:354`). `createOpenAILlm` (`358-367`) and `createOpenAICompatibleLlm` (`374-398`) construct `ChatOpenAI` with **no `maxTokens` at all**. Both `openai-compatible` and legacy `ollama` route through `createOpenAICompatibleLlm` (`chain.ts:369`, `436-451`).

### Failure mode (thinking models on an OpenAI-compatible endpoint)

A "thinking"/reasoning model (e.g. `gemma4-*` served behind an OpenAI-compatible endpoint) emits its reasoning into a separate thinking/reasoning field that still counts against the response budget. With **no `maxTokens` cap**, the server applies its own default budget, and the model can spend that entire budget on reasoning tokens. The stream then hits `finish_reason=length` before any `content` is produced, so the assistant `content` comes back **empty**.

Operational symptom: `oracle_query` (or the CLI answer path) returns a **blank answer** rather than an error. Retrieval succeeded, sources are present, but the synthesized answer text is empty. It does not look like a failure — it looks like the model had nothing to say. The Anthropic lane does not exhibit this because its explicit `maxTokens: 4096` reserves room for `content`.

Practical mitigations when you hit blank answers on this lane: set a server-side or model-side response/output-token limit high enough to leave room after reasoning, disable the model's thinking mode, or point `ORACLE_LLM_MODEL` at a non-thinking model.

### This repo is currently on the gotcha lane

This repo's own `.env` (gitignored, so not listed in this doc's `sources:`) runs exactly the un-capped, thinking-model configuration, as of this verification:

```
ORACLE_LLM_PROVIDER=openai-compatible
ORACLE_LLM_BASE_URL=http://100.113.49.37:11434/v1
ORACLE_LLM_API_KEY=ollama
ORACLE_LLM_MODEL=gemma4-26b-a4b-64k
```

(`ORACLE_EMBEDDING_PROVIDER=openai` in the same file — a concrete instance of the two-lane split.) So blank-answer reports against this checkout should suspect the token-budget asymmetry first.

## Other LLM knobs

- `ORACLE_LLM_BASE_URL` -> `config.llmBaseUrl` (`config.ts:139`), `ORACLE_LLM_API_KEY` -> `config.llmApiKey` (`140`), `ORACLE_LLM_MODEL` -> `config.llmModel` (`137`). These are the preferred `openai-compatible` inputs.
- `createOpenAICompatibleLlm` resolves the base URL through an intermediate `ollamaBaseUrl` local: `config.ollamaBaseUrl`, falling back to `DEFAULT_OLLAMA_BASE_URL` only for the legacy `ollama` alias (`chain.ts:382-383`), then `config.llmBaseUrl ?? ollamaBaseUrl!` (`chain.ts:384`). This indirection is the `ab6aad16` fix: `config.ollamaBaseUrl` no longer carries a schema default, so an `openai-compatible` lane with nothing configured genuinely resolves `ollamaBaseUrl` to `undefined` here instead of silently landing on localhost. The key resolves as `config.llmApiKey ?? config.ollamaApiKey ?? fallbackKey` (`chain.ts:391`), where `fallbackKey` is `"ollama"` for the legacy alias and `""` otherwise (`chain.ts:390`). It deliberately does **not** fall back to `openaiApiKey` (`chain.ts:372-373`).
- Ollama base-url precedence (`config.ts:134`): `ORACLE_OLLAMA_BASE_URL ?? OLLAMA_BASE_URL`, with no schema default anymore (`config.ts:47`). The legacy localhost default (`DEFAULT_OLLAMA_BASE_URL`, `config.ts:8`) is applied only at the `ollama`-alias call sites (`chain.ts` LLM branch, `embeddings.ts` embedding branch), so an unset `ollamaBaseUrl` reaches the `openai-compatible` branch's `!config.llmBaseUrl && !config.ollamaBaseUrl` guard as a real `undefined` and correctly throws (fixed; previously dead code — see task ab6aad16).
- `ORACLE_LLM_PROVIDER=ollama` is deprecated in favor of `openai-compatible` + `ORACLE_LLM_BASE_URL`/`ORACLE_LLM_API_KEY`; it prints a one-time warning (`chain.ts:401-411`).

## Authoritative env reference

For the full env-var table, see [../configuration.md](../configuration.md). That doc does **not** cover the two-enum distinction or the `maxTokens` asymmetry — this doc exists to state exactly those two things. Do not duplicate its table here.
