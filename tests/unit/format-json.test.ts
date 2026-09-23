import { describe, expect, it } from "vitest";
import { formatSearchJson, formatReposJson, formatQueryJson } from "../../src/format-json.js";
import { Document } from "@langchain/core/documents";

describe("formatQueryJson", () => {
  it("serializes the stable query schema without invoking an LLM", () => {
    expect(
      JSON.parse(
        formatQueryJson("where?", {
          answer: "Here.",
          sources: [
            { repo: "demo", filePath: "src/a.ts", snippet: "ignored" },
          ],
          pointers: ["demo/docs/design.md"],
        }),
      ),
    ).toEqual({
      ok: true,
      question: "where?",
      answer: "Here.",
      sources: [{ filePath: "src/a.ts", repo: "demo" }],
      pointers: ["demo/docs/design.md"],
    });
  });

  it("adds degraded and degradedReason only when the result is degraded", () => {
    expect(
      JSON.parse(
        formatQueryJson("where?", {
          answer: "LLM request failed. Returning raw retrieved context instead.\n\n...",
          sources: [],
          pointers: [],
          degraded: true,
          degradedReason: "llm_request_failed",
        }),
      ),
    ).toEqual({
      ok: true,
      question: "where?",
      answer: "LLM request failed. Returning raw retrieved context instead.\n\n...",
      sources: [],
      pointers: [],
      degraded: true,
      degradedReason: "llm_request_failed",
    });
  });

  it("omits degraded and degradedReason entirely on a non-degraded result", () => {
    const parsed = JSON.parse(
      formatQueryJson("where?", {
        answer: "Here.",
        sources: [],
        pointers: [],
      }),
    );
    expect(parsed).not.toHaveProperty("degraded");
    expect(parsed).not.toHaveProperty("degradedReason");
  });
});

describe("formatSearchJson", () => {
  it("marks a successful search document ok: true", () => {
    const doc = new Document({ pageContent: "x", metadata: { repo: "r", filePath: "a.ts" } });
    const parsed = JSON.parse(formatSearchJson("q", "r", 5, [doc]));
    expect(parsed.ok).toBe(true);
  });
});

describe("formatReposJson", () => {
  it("marks a successful list-repos document ok: true", () => {
    const parsed = JSON.parse(formatReposJson([]));
    expect(parsed).toEqual({ ok: true, repos: [] });
  });
});
