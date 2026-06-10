import { createRequire } from "node:module";

// Read the version once from package.json so the CLI (--version), the MCP
// handshake, and the HTTP /health surface never drift from each other or from
// the published package. package.json is the single source of truth: a
// release-prep PR only has to bump it (+ package-lock.json).
const require = createRequire(import.meta.url);
const pkg = require("../package.json") as { version: string };

export const VERSION = pkg.version;
