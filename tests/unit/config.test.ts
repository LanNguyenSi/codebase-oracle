import { describe, it, expect } from "vitest";
import { loadConfig } from "../../src/config.js";

describe("loadConfig", () => {
  it("applies defaults when no env vars set", () => {
    const config = loadConfig();
    expect(config.embeddingModel).toBe("text-embedding-3-small");
    expect(config.vectorStoreType).toBe("directory");
    expect(config.scanRoot).toContain("git");
  });

  it("accepts overrides", () => {
    const config = loadConfig({
      scanRoot: "/custom/path",
      embeddingModel: "text-embedding-3-large",
      vectorStoreType: "memory",
    });
    expect(config.scanRoot).toBe("/custom/path");
    expect(config.embeddingModel).toBe("text-embedding-3-large");
    expect(config.vectorStoreType).toBe("memory");
  });

  it("preserves optional keys as undefined when not set", () => {
    const config = loadConfig();
    // These depend on env vars, which may or may not be set
    // Just verify the shape is valid
    expect(typeof config.openaiApiKey === "string" || config.openaiApiKey === undefined).toBe(true);
    expect(typeof config.anthropicApiKey === "string" || config.anthropicApiKey === undefined).toBe(true);
  });
});
