import { describe, it, expect } from "vitest";
import { loadConfig } from "../../src/config.js";

describe("loadConfig", () => {
  it("applies defaults for optional fields", () => {
    const config = loadConfig({ scanRoot: "/tmp/repos" });
    expect(config.embeddingProvider).toBe("openai");
    expect(config.llmProvider).toBe("auto");
    expect(config.embeddingModel).toBe("text-embedding-3-small");
    expect(config.llmModel).toBe("claude-sonnet-4-20250514");
    expect(config.ollamaBaseUrl).toBe("http://localhost:11434/v1");
    expect(config.vectorStoreType).toBe("directory");
  });

  it("throws when scanRoot is not provided", () => {
    expect(() => loadConfig()).toThrow();
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
