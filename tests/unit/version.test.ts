import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { VERSION } from "../../src/version.js";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "../..");
const pkg = JSON.parse(readFileSync(resolve(repoRoot, "package.json"), "utf8")) as {
  version: string;
};

describe("version", () => {
  it("VERSION is read from package.json (single source of truth)", () => {
    expect(VERSION).toBe(pkg.version);
  });

  it("CLI / MCP / HTTP entrypoints use the shared VERSION constant, not a hardcoded literal", () => {
    for (const file of ["index.ts", "mcp-server.ts", "http-server.ts"]) {
      const src = readFileSync(resolve(repoRoot, "src", file), "utf8");
      // They must import + reference the shared constant...
      expect(src).toMatch(/import \{ VERSION \} from "\.\/version\.js"/);
      expect(src).toMatch(/\bVERSION\b/);
      // ...and must not re-hardcode a semver literal that could drift from package.json.
      expect(src).not.toMatch(/version:\s*["']\d+\.\d+\.\d+["']/);
      expect(src).not.toMatch(/\.version\(\s*["']\d+\.\d+\.\d+["']\s*\)/);
    }
  });
});
