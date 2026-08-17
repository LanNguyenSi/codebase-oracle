import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    coverage: {
      provider: "v8",
      // Whole-src: every matched file is reported, so a NEW untested file lowers
      // coverage below the floor (not just erosion of already-tested files).
      include: ["src/**/*.ts"],
      exclude: ["src/**/*.d.ts"],
      reporter: ["text-summary"],
      // Ratchet raised on 2026-08-17 (http-server.ts 500-path + releasePair
      // coverage added), locked a few points below the newly measured
      // baseline (lines 78.0 / stmts 76.4 / funcs 77.1 / branches 75.7,
      // stable across repeated runs). Previous floor (2026-06-29 baseline:
      // lines 71.8 / stmts 70.3 / funcs 70.7 / branches 70.3) was 67/65/65/65.
      // Raise further as coverage grows.
      thresholds: {
        lines: 74,
        statements: 73,
        functions: 74,
        branches: 72,
      },
    },
  },
});
