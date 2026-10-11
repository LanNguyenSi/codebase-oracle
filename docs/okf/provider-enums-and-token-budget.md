---
type: invariant
title: Two provider enums, and the token budget only one of them sets
description: embeddingProvider and llmProvider are independent enums with independent env vars; only the Anthropic LLM lane caps maxTokens, so an uncapped OpenAI-compatible thinking model can return empty content.
tags: [config, providers, llm, embeddings, gotcha]
timestamp: 2026-10-06T04:08:19Z
sources:
  - src/config.ts
  - src/retrieval/chain.ts
  - src/store/embeddings.ts
  - docs/configuration.md
---

## Invariant

Provider selection in codebase-oracle is split across **two separate, independently-set enums**. They are routinely conflated because both are "the provider," but embedding and answering are different lanes with different env vars, different default models, and different runtime construction. One of those lanes (Anthropic) caps the response token budget; the other two (OpenAI, OpenAI-compatible) do not, and that asymmetry is a real failure mode, described below.

## The two enums (`src/config.ts:25-32#" ]).default("`)

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

- `embeddingProvider`, env `ORACLE_EMBEDDING_PROVIDER` (`config.ts:130-131#"process.env.ORACLE_EMBEDDING_PROVIDER"`). Values: `openai | ollama | stub`. (`stub` = deterministic hash vectors, integration tests only, `config.ts:22-24#"tests/integration/index-cli.test.ts"`.)
- `llmProvider`, env `ORACLE_LLM_PROVIDER` (`config.ts:133-134#"process.env.ORACLE_LLM_PROVIDER"`). Values: `auto | anthropic | openai | openai-compatible | ollama`.

They are **independent**: you can embed with `openai` and answer with `ollama` (or any other combination). The comment at `config.ts:52-55#"without leaking keys across lanes"` says the `ORACLE_LLM_*` pair is kept separate from `openaiApiKey`/`openaiBaseUrl` because doing so "lets embedding and LLM live on different providers without leaking keys across lanes." That holds for the answer LLM, which never reads `OPENAI_API_KEY` on the `openai-compatible` lane. It does not hold for the `ollama` embedding lane, which falls back to `OPENAI_API_KEY` when `OLLAMA_API_KEY` is unset (see "Which key goes to which endpoint" below).

## Default models (`loadConfig`, `config.ts:137-144#"claude-sonnet-4-6"`)

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

### What `auto` resolves to (`createLlm`, `src/retrieval/chain.ts:480-522#"return null;"`)

`auto` is not routed by a dedicated branch. It falls through the explicit-provider `if`s and lands on the credential-sniffing tail: `anthropicApiKey` present -> Anthropic (`chain.ts:514-515#"return createAnthropicLlm(config);"`); else `openaiApiKey` present -> OpenAI with the hardcoded `OPENAI_AUTO_FALLBACK_MODEL = "gpt-4o-mini"` (`chain.ts:24#"const OPENAI_AUTO_FALLBACK_MODEL"`, `chain.ts:518-519#"OPENAI_AUTO_FALLBACK_MODEL"`); else returns `null` (`chain.ts:522#"return null;"`). `auto` **never** resolves to `openai-compatible` or `ollama`: those require an explicit `ORACLE_LLM_PROVIDER`.

## The gotcha: only the Anthropic lane caps `maxTokens`

Three LLM constructors, three different budget behaviors (`src/retrieval/chain.ts`):

```ts
function createAnthropicLlm(config: Config) {
  return new ChatAnthropic({ ..., temperature: 0, maxTokens: 4096 });
}

function createOpenAILlm(config: Config, modelName: string) {
  return new ChatOpenAI({ ..., temperature: 0, /* NO maxTokens */ });
}

function createOpenAICompatibleLlm(config: Config, isLegacyOllama: boolean) {
  return new ChatOpenAI({ ..., temperature: 0, /* NO maxTokens */ });
}
```

`createAnthropicLlm` (`chain.ts:397-411#"clientOptions: { timeout: config.llmTimeoutMs"`) sets `maxTokens: 4096` (`chain.ts:402#"maxTokens: 4096"`). `createOpenAILlm` (`chain.ts:415-424#": undefined,"`) and `createOpenAICompatibleLlm` (`chain.ts:433-457#"configuration: { baseURL },"`) construct `ChatOpenAI` with **no `maxTokens` at all** (all three also set a shared `timeout`/`maxRetries` pair, orthogonal to the token-budget gotcha below). Both `openai-compatible` and legacy `ollama` route through `createOpenAICompatibleLlm` (`chain.ts:428#"route here"`, `chain.ts:507#"createOpenAICompatibleLlm(config, false)"`, `chain.ts:511#"createOpenAICompatibleLlm(config, true)"`).

### Failure mode (thinking models on an OpenAI-compatible endpoint)

A "thinking"/reasoning model served behind an OpenAI-compatible endpoint emits its reasoning into a separate thinking/reasoning field that still counts against the response budget. With **no `maxTokens` cap**, the server applies its own default budget, and the model can spend that entire budget on reasoning tokens. The stream then hits `finish_reason=length` before any `content` is produced, so the assistant `content` comes back **empty**.

Operational symptom: `oracle_query` (or the CLI answer path) returns a **blank answer** rather than an error. Retrieval succeeded, sources are present, but the synthesized answer text is empty. It does not look like a failure; it looks like the model had nothing to say. The Anthropic lane does not exhibit this because its explicit `maxTokens: 4096` reserves room for `content`.

Practical mitigations when you hit blank answers on this lane: set a server-side or model-side response/output-token limit high enough to leave room after reasoning, disable the model's thinking mode, or point `ORACLE_LLM_MODEL` at a non-thinking model.

### A typical affected setup

A typical affected setup is `ORACLE_LLM_PROVIDER=openai-compatible` pointing at a local OpenAI-compatible endpoint (for example `http://localhost:11434/v1`) that serves a thinking model. When such a setup returns blank answers, suspect the token-budget asymmetry first. The embedding lane is unaffected by this: it is configured separately (`ORACLE_EMBEDDING_PROVIDER`) and can stay on `openai` while the answer LLM runs locally.

## Which key goes to which endpoint

Each lane resolves its own credential and base URL:

- **Embeddings, `openai`**: `OPENAI_API_KEY` (required, the constructor throws without it) to `OPENAI_BASE_URL` when set, else the OpenAI SDK default (`src/store/embeddings.ts:64-76#": undefined,"`).
- **Embeddings, `ollama`**: `OLLAMA_API_KEY`, else `OPENAI_API_KEY`, else the literal `"ollama"`, sent to `ORACLE_OLLAMA_BASE_URL ?? OLLAMA_BASE_URL`, else `DEFAULT_OLLAMA_BASE_URL` (`src/store/embeddings.ts:51-59#"config.ollamaBaseUrl ?? DEFAULT_OLLAMA_BASE_URL"`). Set `OLLAMA_API_KEY` explicitly when `OPENAI_API_KEY` is also set and that key must not reach the Ollama endpoint.
- **LLM, `anthropic`** (and `auto` with `ANTHROPIC_API_KEY`): `ANTHROPIC_API_KEY` to the Anthropic SDK default endpoint (`chain.ts:397-399#"anthropicApiKey: config.anthropicApiKey!"`).
- **LLM, `openai`** (and `auto` with only `OPENAI_API_KEY`): `OPENAI_API_KEY` to `OPENAI_BASE_URL` when set, else the OpenAI SDK default (`chain.ts:415-424#": undefined,"`). This lane shares both values with the `openai` embedding lane.
- **LLM, `openai-compatible` and the legacy `ollama` alias**: `ORACLE_LLM_API_KEY`, else `OLLAMA_API_KEY`, else `""` (or `"ollama"` for the legacy alias), sent to `ORACLE_LLM_BASE_URL`, else `ORACLE_OLLAMA_BASE_URL ?? OLLAMA_BASE_URL`, else (legacy alias only) `DEFAULT_OLLAMA_BASE_URL` (`chain.ts:441-450#"const apiKey = config.llmApiKey"`). It never falls back to `OPENAI_API_KEY` (`chain.ts:431-432#"for the LLM was the old footgun"`).

## Other LLM knobs

- `ORACLE_LLM_BASE_URL` -> `config.llmBaseUrl` (`config.ts:159#"ORACLE_LLM_BASE_URL"`), `ORACLE_LLM_API_KEY` -> `config.llmApiKey` (`config.ts:160#"ORACLE_LLM_API_KEY"`), `ORACLE_LLM_MODEL` -> `config.llmModel` (`config.ts:157#"ORACLE_LLM_MODEL"`). These are the preferred `openai-compatible` inputs.
- `createOpenAICompatibleLlm` resolves the base URL through an intermediate `ollamaBaseUrl` local: `config.ollamaBaseUrl`, falling back to `DEFAULT_OLLAMA_BASE_URL` only for the legacy `ollama` alias (`chain.ts:441-442#"isLegacyOllama ? DEFAULT_OLLAMA_BASE_URL : undefined"`), then `config.llmBaseUrl ?? ollamaBaseUrl!` (`chain.ts:443#"config.llmBaseUrl ?? ollamaBaseUrl!"`). Because `config.ollamaBaseUrl` carries no schema default, an `openai-compatible` lane with nothing configured resolves `ollamaBaseUrl` to `undefined` here instead of silently landing on localhost. The key resolves as `config.llmApiKey ?? config.ollamaApiKey ?? fallbackKey` (`chain.ts:450#"config.llmApiKey ?? config.ollamaApiKey ?? fallbackKey"`), where `fallbackKey` is `"ollama"` for the legacy alias and `""` otherwise (`chain.ts:449#"const fallbackKey"`). It deliberately does **not** fall back to `openaiApiKey` (`chain.ts:431-432#"for the LLM was the old footgun"`).
- Ollama base-url precedence (`config.ts:154#"ORACLE_OLLAMA_BASE_URL ?? process.env.OLLAMA_BASE_URL"`): `ORACLE_OLLAMA_BASE_URL ?? OLLAMA_BASE_URL`, with no schema default (`config.ts:47#"ollamaBaseUrl: z.string().optional()"`). The legacy localhost default (`DEFAULT_OLLAMA_BASE_URL`, `config.ts:8#"export const DEFAULT_OLLAMA_BASE_URL"`) is applied only at the `ollama`-alias call sites (`chain.ts` LLM branch, `embeddings.ts` embedding branch), so an unset `ollamaBaseUrl` reaches the `openai-compatible` branch's `!config.llmBaseUrl && !config.ollamaBaseUrl` guard (`chain.ts:502#"!config.llmBaseUrl && !config.ollamaBaseUrl"`) as a real `undefined` and throws.
- `ORACLE_LLM_PROVIDER=ollama` is deprecated in favor of `openai-compatible` + `ORACLE_LLM_BASE_URL`/`ORACLE_LLM_API_KEY`; it prints a one-time warning (`chain.ts:462-470#"still resolve as fallbacks for now."`).
- Not covered here (out of this doc's stated scope): every LLM constructor also sets a shared request timeout and `maxRetries: 0` from `config.llmTimeoutMs` (`ORACLE_LLM_TIMEOUT_MS`), see `../configuration.md` for that env var and its default's reasoning.

## Authoritative env reference

For the full env-var table, see [../configuration.md](../configuration.md). That doc lists both enums in separate rows, their default models, and which key each lane sends, but it does not explain the `maxTokens` asymmetry; this doc covers that. Do not duplicate its table here.
