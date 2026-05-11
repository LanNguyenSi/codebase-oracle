// Environment-variable reader for config-toy. Decouples zod parsing from
// the actual source so tests can swap in a fixed map.

import fs from "node:fs";
import path from "node:path";

export function readEnv(): Record<string, string | undefined> {
  const envFile = path.resolve(process.cwd(), ".env");
  if (fs.existsSync(envFile)) {
    for (const line of fs.readFileSync(envFile, "utf-8").split(/\r?\n/)) {
      const match = line.match(/^([A-Z_][A-Z0-9_]*)=(.*)$/);
      if (match && process.env[match[1]] === undefined) {
        process.env[match[1]] = match[2];
      }
    }
  }
  return { ...process.env };
}
