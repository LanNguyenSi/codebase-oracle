import { afterEach, describe, it, expect } from "vitest";
import { assertScanRoot, loadConfig } from "../../src/config.js";

describe("loadConfig", () => {
  it("applies defaults for optional fields", () => {
    const config = loadConfig({ scanRoot: "/tmp/repos" });
    expect(config.embeddingProvider).toBe("openai");
    expect(config.llmProvider).toBe("auto");
    expect(config.embeddingModel).toBe("text-embedding-3-small");
    expect(config.llmModel).toBe("claude-sonnet-4-6");
    // ollamaBaseUrl deliberately has no schema default (see config.ts): the
    // legacy localhost fallback now lives at the `ollama` alias call sites
    // in retrieval/chain.ts and store/embeddings.ts instead.
    expect(config.ollamaBaseUrl).toBeUndefined();
    expect(config.vectorStoreType).toBe("directory");
  });

  it("loads without scanRoot for read-only commands", () => {
    const prev = process.env.ORACLE_SCAN_ROOT;
    delete process.env.ORACLE_SCAN_ROOT;
    try {
      const config = loadConfig();
      expect(config.scanRoot).toBeUndefined();
      expect(config.embeddingProvider).toBe("openai");
    } finally {
      if (prev !== undefined) process.env.ORACLE_SCAN_ROOT = prev;
    }
  });

  it("accepts overrides", () => {
    const config = loadConfig({
      scanRoot: "/custom/path",
      embeddingProvider: "ollama",
      llmProvider: "ollama",
      ollamaBaseUrl: "http://localhost:11434/v1",
      embeddingModel: "text-embedding-3-large",
      llmModel: "llama3.1",
      vectorStoreType: "memory",
    });
    expect(config.scanRoot).toBe("/custom/path");
    expect(config.embeddingProvider).toBe("ollama");
    expect(config.llmProvider).toBe("ollama");
    expect(config.embeddingModel).toBe("text-embedding-3-large");
    expect(config.llmModel).toBe("llama3.1");
    expect(config.vectorStoreType).toBe("memory");
  });

  it("uses provider-aware model defaults", () => {
    const ollamaConfig = loadConfig({
      scanRoot: "/tmp/repos",
      embeddingProvider: "ollama",
      llmProvider: "ollama",
    });
    expect(ollamaConfig.embeddingModel).toBe("nomic-embed-text");
    expect(ollamaConfig.llmModel).toBe("llama3.1");

    const openAIConfig = loadConfig({
      scanRoot: "/tmp/repos",
      llmProvider: "openai",
    });
    expect(openAIConfig.llmModel).toBe("gpt-4o-mini");
  });

  it("preserves optional keys as undefined when not set", () => {
    const config = loadConfig({ scanRoot: "/tmp/repos" });
    expect(typeof config.openaiApiKey === "string" || config.openaiApiKey === undefined).toBe(true);
    expect(typeof config.openaiBaseUrl === "string" || config.openaiBaseUrl === undefined).toBe(true);
    expect(typeof config.anthropicApiKey === "string" || config.anthropicApiKey === undefined).toBe(true);
    expect(typeof config.ollamaApiKey === "string" || config.ollamaApiKey === undefined).toBe(true);
  });
});

describe("loadConfig maxFileSizeBytes / ORACLE_MAX_FILE_SIZE", () => {
  const prevEnv = process.env.ORACLE_MAX_FILE_SIZE;

  afterEach(() => {
    if (prevEnv === undefined) delete process.env.ORACLE_MAX_FILE_SIZE;
    else process.env.ORACLE_MAX_FILE_SIZE = prevEnv;
  });

  it("defaults to 500_000 when unset", () => {
    delete process.env.ORACLE_MAX_FILE_SIZE;
    const config = loadConfig({ scanRoot: "/tmp/repos" });
    expect(config.maxFileSizeBytes).toBe(500_000);
  });

  it("parses a set env var as an integer", () => {
    process.env.ORACLE_MAX_FILE_SIZE = "1000000";
    const config = loadConfig({ scanRoot: "/tmp/repos" });
    expect(config.maxFileSizeBytes).toBe(1_000_000);
  });

  it("treats an empty string as unset (an `ORACLE_MAX_FILE_SIZE=` .env line must not crash)", () => {
    process.env.ORACLE_MAX_FILE_SIZE = "";
    const config = loadConfig({ scanRoot: "/tmp/repos" });
    expect(config.maxFileSizeBytes).toBe(500_000);
  });

  it.each(["abc", "0", "-5"])(
    "throws for an invalid value (%s) instead of silently falling back",
    (raw) => {
      process.env.ORACLE_MAX_FILE_SIZE = raw;
      expect(() => loadConfig({ scanRoot: "/tmp/repos" })).toThrow();
    },
  );
});

describe("loadConfig maxTextFileSizeBytes / ORACLE_MAX_TEXT_FILE_SIZE", () => {
  const prevEnv = process.env.ORACLE_MAX_TEXT_FILE_SIZE;

  afterEach(() => {
    if (prevEnv === undefined) delete process.env.ORACLE_MAX_TEXT_FILE_SIZE;
    else process.env.ORACLE_MAX_TEXT_FILE_SIZE = prevEnv;
  });

  it("defaults to 2_000_000 when unset", () => {
    delete process.env.ORACLE_MAX_TEXT_FILE_SIZE;
    const config = loadConfig({ scanRoot: "/tmp/repos" });
    expect(config.maxTextFileSizeBytes).toBe(2_000_000);
  });

  it("parses a set env var as an integer", () => {
    process.env.ORACLE_MAX_TEXT_FILE_SIZE = "3000000";
    const config = loadConfig({ scanRoot: "/tmp/repos" });
    expect(config.maxTextFileSizeBytes).toBe(3_000_000);
  });

  it("treats an empty string as unset (an `ORACLE_MAX_TEXT_FILE_SIZE=` .env line must not crash)", () => {
    process.env.ORACLE_MAX_TEXT_FILE_SIZE = "";
    const config = loadConfig({ scanRoot: "/tmp/repos" });
    expect(config.maxTextFileSizeBytes).toBe(2_000_000);
  });

  it.each(["abc", "0", "-5"])(
    "throws for an invalid value (%s) instead of silently falling back",
    (raw) => {
      process.env.ORACLE_MAX_TEXT_FILE_SIZE = raw;
      expect(() => loadConfig({ scanRoot: "/tmp/repos" })).toThrow();
    },
  );

  it("is independent from ORACLE_MAX_FILE_SIZE (setting one does not move the other)", () => {
    const prevGeneral = process.env.ORACLE_MAX_FILE_SIZE;
    try {
      process.env.ORACLE_MAX_FILE_SIZE = "1000";
      delete process.env.ORACLE_MAX_TEXT_FILE_SIZE;
      const config = loadConfig({ scanRoot: "/tmp/repos" });
      expect(config.maxFileSizeBytes).toBe(1000);
      expect(config.maxTextFileSizeBytes).toBe(2_000_000);
    } finally {
      if (prevGeneral === undefined) delete process.env.ORACLE_MAX_FILE_SIZE;
      else process.env.ORACLE_MAX_FILE_SIZE = prevGeneral;
    }
  });
});

describe("assertScanRoot", () => {
  it("throws a friendly error when scanRoot is undefined", () => {
    const prev = process.env.ORACLE_SCAN_ROOT;
    delete process.env.ORACLE_SCAN_ROOT;
    try {
      const config = loadConfig();
      expect(() => assertScanRoot(config)).toThrow(/ORACLE_SCAN_ROOT is required/);
    } finally {
      if (prev !== undefined) process.env.ORACLE_SCAN_ROOT = prev;
    }
  });

  it("passes through when scanRoot is set", () => {
    const config = loadConfig({ scanRoot: "/tmp/repos" });
    expect(() => assertScanRoot(config)).not.toThrow();
  });
});
