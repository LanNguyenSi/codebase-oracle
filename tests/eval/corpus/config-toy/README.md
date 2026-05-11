# config-toy

Tiny zod-based config loader used as eval fixture for codebase-oracle.

The flow:

1. `src/env.ts` reads `process.env` (and a `.env` file if present).
2. `src/config.ts` parses the raw env through a zod schema, returning a
   strongly-typed `Config` value or throwing a descriptive error.

Real-world variants of this pattern live in nearly every Node service we
operate; the eval uses this fixture to exercise oracle's ability to
answer "where are environment variables validated" without picking up
unrelated env-touching code.
