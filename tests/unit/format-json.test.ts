import { describe, expect, it } from "vitest";
import { formatQueryJson } from "../../src/format-json.js";

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
      question: "where?",
      answer: "Here.",
      sources: [{ filePath: "src/a.ts", repo: "demo" }],
      pointers: ["demo/docs/design.md"],
    });
  });
});
