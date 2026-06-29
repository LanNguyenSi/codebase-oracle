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
      // Ratchet locked just below the 2026-06-29 measured baseline
      // (lines 71.8 / stmts 70.3 / funcs 70.7 / branches 70.3). Raise as coverage grows.
      thresholds: {
        lines: 67,
        statements: 65,
        functions: 65,
        branches: 65,
      },
    },
  },
});
