---
type: invariant
title: Two provider enums, and the token budget only one of them sets
description: embeddingProvider and llmProvider are independent enums with independent env vars; only the Anthropic LLM lane caps maxTokens, so an uncapped OpenAI-compatible thinking model can return empty content.
tags: [config, providers, llm, embeddings, gotcha]
timestamp: 2026-07-10T08:34:41.256341Z
sources:
  - src/config.ts
  - src/retrieval/chain.ts
  - docs/configuration.md
---

## Invariant

Provider selection in codebase-oracle is split across **two separate, independently-set enums**. They are routinely conflated because both are "the provider," but embedding and answering are different lanes with different env vars, different default models, and different runtime construction. One of those lanes (Anthropic) caps the response token budget; the other two (OpenAI, OpenAI-compatible) do not — and that asymmetry is a live operational failure mode on this repo's current config.

## The two enums (`src/config.ts:18-25`)

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

- `embeddingProvider` — env `ORACLE_EMBEDDING_PROVIDER` (`config.ts:82-84`). Values: `openai | ollama | stub`. (`stub` = deterministic hash vectors, integration tests only — `config.ts:15-17`.)
- `llmProvider` — env `ORACLE_LLM_PROVIDER` (`config.ts:85-87`). Values: `auto | anthropic | openai | openai-compatible | ollama`.

They are **independent**: you can embed with `openai` and answer with `ollama` (or any other combination). The comment at `config.ts:36-39` states the keys are deliberately kept separate "so embedding and LLM live on different providers without leaking keys across lanes."

## Default models (`loadConfig`, `config.ts:89-96`)

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

### What `auto` resolves to (`createLlm`, `src/retrieval/chain.ts:413-452`)

`auto` is not routed by a dedicated branch. It falls through the explicit-provider `if`s and lands on the credential-sniffing tail: `anthropicApiKey` present -> Anthropic (`443-444`); else `openaiApiKey` present -> OpenAI with the hardcoded `OPENAI_AUTO_FALLBACK_MODEL = "gpt-4o-mini"` (`chain.ts:24`, `447-448`); else returns `null` (`451`). `auto` **never** resolves to `openai-compatible` or `ollama` — those require an explicit `ORACLE_LLM_PROVIDER`.

## The gotcha: only the Anthropic lane caps `maxTokens`

Three LLM constructors, three different budget behaviors (`src/retrieval/chain.ts`):

```ts
function createAnthropicLlm(config: Config) {          // 349-356
  return new ChatAnthropic({ ..., temperature: 0, maxTokens: 4096 });
}

function createOpenAILlm(config: Config, modelName: string) {   // 358-367
  return new ChatOpenAI({ ..., temperature: 0, /* NO maxTokens */ });
}

function createOpenAICompatibleLlm(config: Config, isLegacyOllama: boolean) {  // 374-392
  return new ChatOpenAI({ ..., temperature: 0, /* NO maxTokens */ });
}
```

`createAnthropicLlm` sets `maxTokens: 4096` (`chain.ts:354`). `createOpenAILlm` (`358-367`) and `createOpenAICompatibleLlm` (`374-392`) construct `ChatOpenAI` with **no `maxTokens` at all**. Both `openai-compatible` and legacy `ollama` route through `createOpenAICompatibleLlm` (`chain.ts:369`, `430-441`).

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

- `ORACLE_LLM_BASE_URL` -> `config.llmBaseUrl` (`config.ts:111`), `ORACLE_LLM_API_KEY` -> `config.llmApiKey` (`112`), `ORACLE_LLM_MODEL` -> `config.llmModel` (`109`). These are the preferred `openai-compatible` inputs.
- `createOpenAICompatibleLlm` resolves the base URL as `config.llmBaseUrl ?? config.ollamaBaseUrl` and the key as `config.llmApiKey ?? config.ollamaApiKey ?? fallbackKey` (`chain.ts:378`, `384-385`), where `fallbackKey` is `"ollama"` for the legacy alias and `""` otherwise (`384`). It deliberately does **not** fall back to `openaiApiKey` (`chain.ts:372-373`).
- Ollama base-url precedence (`config.ts:122`): `ORACLE_OLLAMA_BASE_URL ?? OLLAMA_BASE_URL`, with no schema default anymore (`config.ts:47`). The legacy localhost default (`DEFAULT_OLLAMA_BASE_URL`, `config.ts:8`) is applied only at the `ollama`-alias call sites (`chain.ts` LLM branch, `embeddings.ts` embedding branch), so an unset `ollamaBaseUrl` reaches the `openai-compatible` branch's `!config.llmBaseUrl && !config.ollamaBaseUrl` guard as a real `undefined` and correctly throws (fixed; previously dead code — see task ab6aad16).
- `ORACLE_LLM_PROVIDER=ollama` is deprecated in favor of `openai-compatible` + `ORACLE_LLM_BASE_URL`/`ORACLE_LLM_API_KEY`; it prints a one-time warning (`chain.ts:394-405`).

## Authoritative env reference

For the full env-var table, see [../configuration.md](../configuration.md). That doc does **not** cover the two-enum distinction or the `maxTokens` asymmetry — this doc exists to state exactly those two things. Do not duplicate its table here.
```

---

No discrepancies. Every asserted fact matched source at the cited line:
- Two enums and their env vars: `src/config.ts:18-25`, `82-87`.
- Default-model logic and `auto` resolution: `config.ts:89-96`; `chain.ts:413-452` (`auto` falls through to credential-sniffing, never picks `openai-compatible`/`ollama`).
- Token-budget asymmetry: `createAnthropicLlm` `maxTokens: 4096` at `chain.ts:354`; `createOpenAILlm` (`358-367`) and `createOpenAICompatibleLlm` (`374-392`) both have no `maxTokens`.
- `.env` is readable and confirms `ORACLE_LLM_PROVIDER=openai-compatible` with `ORACLE_LLM_MODEL=gemma4-26b-a4b-64k` and `ORACLE_EMBEDDING_PROVIDER=openai`.
- Ollama base-url precedence `ORACLE_OLLAMA_BASE_URL ?? OLLAMA_BASE_URL`: `config.ts:106`.
- `docs/configuration.md` exists; the `../configuration.md` link assumes the doc is placed one directory below `docs/` (as instructed).

One nuance worth flagging (surfaced, not a doc error): `openai-compatible` inherits the `claude-sonnet-4-6` default model (still true). The `!config.ollamaBaseUrl` guard at `chain.ts:441` used to be dead code because `ollamaBaseUrl` always carried a schema default; that has since been fixed (task ab6aad16) by dropping the schema default and resolving the legacy localhost fallback only at the `ollama`-alias call sites, so the guard now throws as intended.
