// Strongly-typed configuration for config-toy.
//
// Reads from the environment via env.ts, validates with zod, and returns a
// frozen Config object. Unknown env vars are ignored; missing required
// vars throw with a descriptive message naming the offending key.

import { z } from "zod";
import { readEnv } from "./env.js";

const ConfigSchema = z.object({
  DATABASE_URL: z.string().url(),
  PORT: z.coerce.number().int().positive().default(3000),
  LOG_LEVEL: z.enum(["debug", "info", "warn", "error"]).default("info"),
  FEATURE_FLAG_BETA: z.coerce.boolean().default(false),
});

export type Config = z.infer<typeof ConfigSchema>;

export function loadConfig(): Config {
  const raw = readEnv();
  const parsed = ConfigSchema.safeParse(raw);
  if (!parsed.success) {
    const issues = parsed.error.issues
      .map((i) => `${i.path.join(".")}: ${i.message}`)
      .join("; ");
    throw new Error(`Invalid environment configuration: ${issues}`);
  }
  return Object.freeze(parsed.data);
}
