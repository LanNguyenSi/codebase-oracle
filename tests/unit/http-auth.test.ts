import { describe, it, expect } from "vitest";
import { resolveHttpBindConfig, verifyBearer } from "../../src/http-auth.js";

describe("resolveHttpBindConfig", () => {
  it("defaults to 127.0.0.1 with no token when nothing is set", () => {
    const cfg = resolveHttpBindConfig({});
    expect(cfg.bind).toBe("127.0.0.1");
    expect(cfg.token).toBeNull();
  });

  it("propagates ORACLE_HTTP_TOKEN when set (loopback bind)", () => {
    const cfg = resolveHttpBindConfig({ ORACLE_HTTP_TOKEN: "secret-token" });
    expect(cfg.bind).toBe("127.0.0.1");
    expect(cfg.token).toBe("secret-token");
  });

  it("treats an empty-string token as unset", () => {
    const cfg = resolveHttpBindConfig({ ORACLE_HTTP_TOKEN: "" });
    expect(cfg.token).toBeNull();
  });

  it("allows ORACLE_HTTP_BIND=0.0.0.0 when a token is also set", () => {
    const cfg = resolveHttpBindConfig({
      ORACLE_HTTP_BIND: "0.0.0.0",
      ORACLE_HTTP_TOKEN: "secret-token",
    });
    expect(cfg.bind).toBe("0.0.0.0");
    expect(cfg.token).toBe("secret-token");
  });

  it("refuses ORACLE_HTTP_BIND=0.0.0.0 without a token", () => {
    expect(() => resolveHttpBindConfig({ ORACLE_HTTP_BIND: "0.0.0.0" })).toThrow(
      /requires ORACLE_HTTP_TOKEN/,
    );
  });

  it("refuses ORACLE_HTTP_BIND=0.0.0.0 with an empty-string token", () => {
    expect(() =>
      resolveHttpBindConfig({ ORACLE_HTTP_BIND: "0.0.0.0", ORACLE_HTTP_TOKEN: "" }),
    ).toThrow(/requires ORACLE_HTTP_TOKEN/);
  });

  it("refuses any off-loopback bind address (LAN IPs) without a token", () => {
    expect(() =>
      resolveHttpBindConfig({ ORACLE_HTTP_BIND: "192.168.1.10" }),
    ).toThrow(/requires ORACLE_HTTP_TOKEN/);
    expect(() =>
      resolveHttpBindConfig({ ORACLE_HTTP_BIND: "::" }),
    ).toThrow(/requires ORACLE_HTTP_TOKEN/);
  });

  it("permits ORACLE_HTTP_BIND=localhost without a token (treated as loopback)", () => {
    const cfg = resolveHttpBindConfig({ ORACLE_HTTP_BIND: "localhost" });
    expect(cfg.bind).toBe("localhost");
    expect(cfg.token).toBeNull();
  });

  it("permits ORACLE_HTTP_BIND=::1 without a token (IPv6 loopback)", () => {
    const cfg = resolveHttpBindConfig({ ORACLE_HTTP_BIND: "::1" });
    expect(cfg.bind).toBe("::1");
    expect(cfg.token).toBeNull();
  });
});

describe("verifyBearer", () => {
  const TOKEN = "s3cret-abc";

  it("accepts any request when no expected token is configured", () => {
    expect(verifyBearer(undefined, null)).toEqual({ ok: true });
    expect(verifyBearer("Bearer whatever", null)).toEqual({ ok: true });
    expect(verifyBearer("", null)).toEqual({ ok: true });
  });

  it("accepts the exact token with 'Bearer' prefix", () => {
    expect(verifyBearer(`Bearer ${TOKEN}`, TOKEN)).toEqual({ ok: true });
  });

  it("accepts 'bearer' in any case", () => {
    expect(verifyBearer(`bearer ${TOKEN}`, TOKEN)).toEqual({ ok: true });
    expect(verifyBearer(`BEARER ${TOKEN}`, TOKEN)).toEqual({ ok: true });
  });

  it("reports 'missing' when the header is absent", () => {
    expect(verifyBearer(undefined, TOKEN)).toEqual({ ok: false, reason: "missing" });
    expect(verifyBearer("", TOKEN)).toEqual({ ok: false, reason: "missing" });
  });

  it("reports 'missing' when a non-Bearer scheme is used", () => {
    expect(verifyBearer("Basic dXNlcjpwYXNz", TOKEN)).toEqual({
      ok: false,
      reason: "missing",
    });
  });

  it("reports 'missing' when the bearer value is empty", () => {
    expect(verifyBearer("Bearer ", TOKEN)).toEqual({ ok: false, reason: "missing" });
  });

  it("reports 'invalid' for a wrong token of the same length", () => {
    const wrong = "s3cret-xyz"; // same length
    expect(verifyBearer(`Bearer ${wrong}`, TOKEN)).toEqual({
      ok: false,
      reason: "invalid",
    });
  });

  it("reports 'invalid' for a wrong token of different length", () => {
    expect(verifyBearer("Bearer short", TOKEN)).toEqual({
      ok: false,
      reason: "invalid",
    });
    expect(verifyBearer(`Bearer ${TOKEN}extra`, TOKEN)).toEqual({
      ok: false,
      reason: "invalid",
    });
  });

  it("does not leak the expected token on a failed compare (defense)", () => {
    // Not a direct timing test, but ensures we never expose the expected
    // token in the AuthResult — callers only see "missing" | "invalid".
    const result = verifyBearer("Bearer nope", TOKEN);
    expect(JSON.stringify(result)).not.toContain(TOKEN);
  });
});
