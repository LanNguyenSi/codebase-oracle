import { z } from "zod";

const configSchema = z.object({
  // Paths
  scanRoot: z.string().default(process.env.HOME + "/git"),
  dataDir: z.string().default(process.env.HOME + "/.codebase-oracle"),

  // Embeddings
  openaiApiKey: z.string().optional(),
  embeddingModel: z.string().default("text-embedding-3-small"),

  // LLM for answer generation
  anthropicApiKey: z.string().optional(),
  llmModel: z.string().default("claude-sonnet-4-20250514"),

  // Vector store
  vectorStoreType: z.enum(["memory", "directory"]).default("directory"),
});

export type Config = z.infer<typeof configSchema>;

export function loadConfig(overrides: Partial<Config> = {}): Config {
  return configSchema.parse({
    scanRoot: process.env.ORACLE_SCAN_ROOT,
    dataDir: process.env.ORACLE_DATA_DIR,
    openaiApiKey: process.env.OPENAI_API_KEY,
    embeddingModel: process.env.ORACLE_EMBEDDING_MODEL,
    anthropicApiKey: process.env.ANTHROPIC_API_KEY,
    llmModel: process.env.ORACLE_LLM_MODEL,
    vectorStoreType: process.env.ORACLE_VECTOR_STORE,
    ...overrides,
  });
}
