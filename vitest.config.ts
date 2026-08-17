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
      // Ratchet raised on 2026-08-17, locked a few points below the newly
      // measured baseline (lines 78.0 / stmts 76.4 / funcs 77.1 / branches
      // 75.7, stable across repeated runs). The floor mostly absorbs
      // coverage that had already accumulated on origin/master since the
      // 2026-06-29 baseline (lines 71.8 / stmts 70.3 / funcs 70.7 / branches
      // 70.3, floor 67/65/65/65 back then): origin/master measured
      // 77.60/76.08/77.09/75.37 BEFORE this task's four new http-server.ts
      // tests were added; those four tests contribute only ~+0.35pp
      // (lines/stmts/branches) and +0.00pp (functions) on top of that.
      // functions is floored at 72, not 74: at the measured 175/227 covered
      // functions, a 74 floor would only tolerate ~9 more untested
      // functions going forward; 72 restores roughly the same ~5pp buffer
      // below measured that the other three metrics keep. Raise further as
      // coverage grows.
      thresholds: {
        lines: 74,
        statements: 73,
        functions: 72,
        branches: 72,
      },
    },
  },
});
