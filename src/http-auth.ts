import { timingSafeEqual } from "node:crypto";

export interface HttpBindConfig {
  bind: string;
  token: string | null;
}

export type AuthResult = { ok: true } | { ok: false; reason: "missing" | "invalid" };

const DEFAULT_BIND = "127.0.0.1";

export function resolveHttpBindConfig(env: NodeJS.ProcessEnv = process.env): HttpBindConfig {
  const rawToken = env.ORACLE_HTTP_TOKEN;
  const token = typeof rawToken === "string" && rawToken.length > 0 ? rawToken : null;
  const bind = env.ORACLE_HTTP_BIND && env.ORACLE_HTTP_BIND.length > 0
    ? env.ORACLE_HTTP_BIND
    : DEFAULT_BIND;

  const loopbackAllowlist = new Set([DEFAULT_BIND, "localhost", "::1"]);
  if (!loopbackAllowlist.has(bind) && !token) {
    throw new Error(
      `ORACLE_HTTP_BIND=${bind} requires ORACLE_HTTP_TOKEN to be set. ` +
        `Binding off-loopback without auth would expose the full indexed corpus. ` +
        `Set a non-empty ORACLE_HTTP_TOKEN, or remove ORACLE_HTTP_BIND to fall back to ${DEFAULT_BIND}.`,
    );
  }

  return { bind, token };
}

export function verifyBearer(
  headerValue: string | undefined,
  expectedToken: string | null,
): AuthResult {
  if (expectedToken === null) return { ok: true };
  if (typeof headerValue !== "string" || headerValue.length === 0) {
    return { ok: false, reason: "missing" };
  }

  const match = headerValue.match(/^Bearer\s+(.+)$/i);
  if (!match) return { ok: false, reason: "missing" };
  const presented = match[1].trim();
  if (presented.length === 0) return { ok: false, reason: "missing" };

  const a = Buffer.from(presented, "utf8");
  const b = Buffer.from(expectedToken, "utf8");
  if (a.length !== b.length) return { ok: false, reason: "invalid" };
  return timingSafeEqual(a, b) ? { ok: true } : { ok: false, reason: "invalid" };
}
