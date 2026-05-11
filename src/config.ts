import { z } from "zod";

const configSchema = z.object({
  // Paths
  scanRoot: z.string().min(1, "ORACLE_SCAN_ROOT is required — set it to the directory containing your git repos"),
  dataDir: z.string().default(process.env.HOME + "/.codebase-oracle"),

  // Provider selection
  // "stub" produces deterministic vectors from a hash of the input text and
  // is intended for integration tests only — it is documented in
  // tests/integration/index-cli.test.ts.
  embeddingProvider: z.enum(["openai", "ollama", "stub"]).default("openai"),
  llmProvider: z.enum(["auto", "anthropic", "openai", "ollama"]).default("auto"),

  // Embeddings
  openaiApiKey: z.string().optional(),
  openaiBaseUrl: z.string().optional(),
  ollamaApiKey: z.string().optional(),
  ollamaBaseUrl: z.string().default("http://localhost:11434/v1"),
  embeddingModel: z.string().default("text-embedding-3-small"),

  // LLM for answer generation
  anthropicApiKey: z.string().optional(),
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
});

export type Config = z.infer<typeof configSchema>;

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
    includeExtensions: parseExtensionsList(process.env.ORACLE_INCLUDE_EXTENSIONS),
    skipDirs: parseCsvList(process.env.ORACLE_SKIP_DIRS),
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
