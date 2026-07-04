import { z } from "zod";

const configSchema = z.object({
  // Paths
  // scanRoot is only consumed by the indexer code path: the `index` and
  // `watch` commands, plus the MCP `oracle_reindex` tool, which all run
  // runIndex -> assertScanRoot. Read-only commands (query, search,
  // list-repos) never touch it, so it stays optional in the schema and is
  // enforced via assertScanRoot at the entry of the writer code paths
  // instead.
  scanRoot: z.string().min(1).optional(),
  dataDir: z.string().default(process.env.HOME + "/.codebase-oracle"),

  // Provider selection
  // "stub" produces deterministic vectors from a hash of the input text and
  // is intended for integration tests only — it is documented in
  // tests/integration/index-cli.test.ts.
  embeddingProvider: z.enum(["openai", "ollama", "stub"]).default("openai"),
  llmProvider: z.enum([
    "auto",
    "anthropic",
    "openai",
    "openai-compatible",
    "ollama",
  ]).default("auto"),

  // Embeddings
  openaiApiKey: z.string().optional(),
  openaiBaseUrl: z.string().optional(),
  ollamaApiKey: z.string().optional(),
  ollamaBaseUrl: z.string().default("http://localhost:11434/v1"),
  embeddingModel: z.string().default("text-embedding-3-small"),

  // LLM for answer generation
  anthropicApiKey: z.string().optional(),
  // Generic OpenAI-compatible LLM endpoint (Groq, Together, OpenRouter,
  // local/cloud Ollama, vLLM, etc.). Keeping these separate from
  // `openaiApiKey`/`openaiBaseUrl` lets embedding and LLM live on different
  // providers without leaking keys across lanes.
  llmBaseUrl: z.string().optional(),
  llmApiKey: z.string().optional(),
  llmModel: z.string().default("claude-sonnet-4-6"),

  // Vector store
  vectorStoreType: z.enum(["memory", "directory"]).default("directory"),

  // Ingest — optional override of the file-extension allowlist
  includeExtensions: z.array(z.string().startsWith(".")).optional(),

  // Ingest — extra directory names to skip on top of the built-in defaults
  // (node_modules, .git, dist, .bun, .opencode-home, etc). Append-only:
  // never replaces the defaults, so forgetting `node_modules` here can't
  // explode the index.
  skipDirs: z.array(z.string().min(1)).optional(),

  // Ingest — per-file size ceiling in bytes. Files larger than this are
  // skipped (loudly reported, never silently dropped — see scanner.ts).
  // An unset env var falls back to the default; an env var present but
  // unparseable/non-positive fails the zod parse instead of silently
  // falling back, since a typo'd limit that resolves to some other
  // default would look like a working config while quietly changing
  // ingest behavior.
  maxFileSizeBytes: z.number().int().positive().default(500_000),
});

export type Config = z.infer<typeof configSchema>;

// Narrows Config so the writer paths can rely on scanRoot being a string.
// Called at the top of runIndex / runWatchMode; throws the same friendly
// message the schema used to emit when scanRoot was strictly required.
export function assertScanRoot(
  config: Config,
): asserts config is Config & { scanRoot: string } {
  if (!config.scanRoot) {
    throw new Error(
      "ORACLE_SCAN_ROOT is required — set it to the directory containing your git repos",
    );
  }
}

export function loadConfig(overrides: Partial<Config> = {}): Config {
  const embeddingProvider = overrides.embeddingProvider
    ?? process.env.ORACLE_EMBEDDING_PROVIDER
    ?? "openai";
  const llmProvider = overrides.llmProvider
    ?? process.env.ORACLE_LLM_PROVIDER
    ?? "auto";

  const defaultEmbeddingModel = embeddingProvider === "ollama"
    ? "nomic-embed-text"
    : "text-embedding-3-small";
  const defaultLlmModel = llmProvider === "openai"
    ? "gpt-4o-mini"
    : llmProvider === "ollama"
      ? "llama3.1"
      : "claude-sonnet-4-6";

  return configSchema.parse({
    scanRoot: process.env.ORACLE_SCAN_ROOT,
    dataDir: process.env.ORACLE_DATA_DIR,
    embeddingProvider,
    llmProvider,
    openaiApiKey: process.env.OPENAI_API_KEY,
    openaiBaseUrl: process.env.OPENAI_BASE_URL,
    ollamaApiKey: process.env.OLLAMA_API_KEY,
    ollamaBaseUrl: process.env.ORACLE_OLLAMA_BASE_URL ?? process.env.OLLAMA_BASE_URL,
    embeddingModel: process.env.ORACLE_EMBEDDING_MODEL ?? defaultEmbeddingModel,
    anthropicApiKey: process.env.ANTHROPIC_API_KEY,
    llmModel: process.env.ORACLE_LLM_MODEL ?? defaultLlmModel,
    vectorStoreType: process.env.ORACLE_VECTOR_STORE,
    llmBaseUrl: process.env.ORACLE_LLM_BASE_URL,
    llmApiKey: process.env.ORACLE_LLM_API_KEY,
    includeExtensions: parseExtensionsList(process.env.ORACLE_INCLUDE_EXTENSIONS),
    skipDirs: parseCsvList(process.env.ORACLE_SKIP_DIRS),
    maxFileSizeBytes: parseMaxFileSizeBytes(process.env.ORACLE_MAX_FILE_SIZE),
    ...overrides,
  });
}

function parseExtensionsList(raw: string | undefined): string[] | undefined {
  if (!raw) return undefined;
  const parts = raw
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean)
    .map((s) => (s.startsWith(".") ? s : `.${s}`))
    .filter((s) => s.length > 1);
  return parts.length > 0 ? parts : undefined;
}

function parseCsvList(raw: string | undefined): string[] | undefined {
  if (!raw) return undefined;
  const parts = raw.split(",").map((s) => s.trim()).filter(Boolean);
  return parts.length > 0 ? parts : undefined;
}

// Unset or empty -> undefined so the schema default (500_000) applies; the
// empty string counts as unset to match parseExtensionsList/parseCsvList
// (an `ORACLE_MAX_FILE_SIZE=` line in a .env template must not crash the
// CLI). A non-empty but unparseable/non-positive value is deliberately NOT
// coerced to a fallback here: it is handed to configSchema.parse below,
// where `.int().positive()` rejects it (NaN, 0, negative all fail) and
// loadConfig throws. A typo'd ORACLE_MAX_FILE_SIZE must fail loudly, not
// silently resolve to some other limit.
function parseMaxFileSizeBytes(raw: string | undefined): number | undefined {
  if (raw === undefined || raw.trim() === "") return undefined;
  return Number(raw);
}
