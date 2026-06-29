import { describe, it, expect, vi, afterEach } from "vitest";
import { mkdtemp, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

// Reset module cache so each test gets a fresh `didLoadEnv = false` flag.
async function freshEnv(): Promise<{ loadEnvFromFile: (filePath?: string) => void }> {
  vi.resetModules();
  // Dynamic import picks up the freshly cleared module registry.
  return import("../../src/env.js") as Promise<{
    loadEnvFromFile: (filePath?: string) => void;
  }>;
}

describe("loadEnvFromFile", () => {
  let tmpDir = "";
  const addedKeys: string[] = [];

  afterEach(async () => {
    if (tmpDir) {
      await rm(tmpDir, { recursive: true, force: true });
      tmpDir = "";
    }
    for (const key of addedKeys) {
      delete process.env[key];
    }
    addedKeys.length = 0;
    vi.resetModules();
  });

  async function makeTmpEnvFile(content: string): Promise<string> {
    if (!tmpDir) {
      tmpDir = await mkdtemp(join(tmpdir(), "oracle-env-test-"));
    }
    const file = join(tmpDir, ".env");
    await writeFile(file, content, "utf8");
    return file;
  }

  it("sets process.env.KEY when the .env file contains KEY=value and the key is unset", async () => {
    addedKeys.push("ORACLE_TEST_ENV_A");
    delete process.env.ORACLE_TEST_ENV_A;
    const file = await makeTmpEnvFile("ORACLE_TEST_ENV_A=hello\n");
    const { loadEnvFromFile } = await freshEnv();
    loadEnvFromFile(file);
    expect(process.env.ORACLE_TEST_ENV_A).toBe("hello");
  });

  it("does NOT override a key already set in process.env (credential-safety precedence guard)", async () => {
    addedKeys.push("ORACLE_TEST_ENV_B");
    process.env.ORACLE_TEST_ENV_B = "already-set";
    const file = await makeTmpEnvFile("ORACLE_TEST_ENV_B=from-file\n");
    const { loadEnvFromFile } = await freshEnv();
    loadEnvFromFile(file);
    expect(process.env.ORACLE_TEST_ENV_B).toBe("already-set");
  });

  it("strips double quotes from values", async () => {
    addedKeys.push("ORACLE_TEST_ENV_C");
    delete process.env.ORACLE_TEST_ENV_C;
    const file = await makeTmpEnvFile('ORACLE_TEST_ENV_C="quoted"\n');
    const { loadEnvFromFile } = await freshEnv();
    loadEnvFromFile(file);
    expect(process.env.ORACLE_TEST_ENV_C).toBe("quoted");
  });

  it("strips single quotes from values", async () => {
    addedKeys.push("ORACLE_TEST_ENV_D");
    delete process.env.ORACLE_TEST_ENV_D;
    const file = await makeTmpEnvFile("ORACLE_TEST_ENV_D='quoted'\n");
    const { loadEnvFromFile } = await freshEnv();
    loadEnvFromFile(file);
    expect(process.env.ORACLE_TEST_ENV_D).toBe("quoted");
  });

  it("skips comment lines (#...), blank lines, and malformed lines (no `=`)", async () => {
    addedKeys.push("ORACLE_TEST_ENV_E");
    delete process.env.ORACLE_TEST_ENV_E;
    const content = [
      "# This is a comment",
      "",
      "MALFORMED_LINE_NO_EQUALS",
      "ORACLE_TEST_ENV_E=valid",
    ].join("\n");
    const file = await makeTmpEnvFile(content);
    const { loadEnvFromFile } = await freshEnv();
    loadEnvFromFile(file);
    expect(process.env.ORACLE_TEST_ENV_E).toBe("valid");
    // Confirm the malformed line did not inject any key
    expect(process.env["MALFORMED_LINE_NO_EQUALS"]).toBeUndefined();
  });

  it("does not throw when the .env file does not exist (missing file is a no-op)", async () => {
    tmpDir = await mkdtemp(join(tmpdir(), "oracle-env-test-"));
    const { loadEnvFromFile } = await freshEnv();
    expect(() => loadEnvFromFile(join(tmpDir, "nonexistent.env"))).not.toThrow();
  });
});
