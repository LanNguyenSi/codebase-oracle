import { afterEach, describe, it, expect } from "vitest";
import { mkdtemp, rm, writeFile, mkdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

// Spawns the real `tsx src/index.ts search` command against a seeded store to
// exercise sources-expansion end-to-end through the CLI: the `[expanded
// from <basename>]` marker printed by the inline render loop, and the
// `--no-expand-sources` opt-out. Follows tests/integration/index-cli.test.ts's
// pattern (spawnSync + stub embedding provider).
//
// The seeded corpus deliberately uses `--path-glob` to pin the ORGANIC result
// list to exactly the frontmatter doc (`**/design.md`), independent of the
// stub embedding provider's hash-based (non-semantic) similarity ordering.
// That keeps the scenario fully deterministic: the pointed-at implementation
// file (`src/impl.ts`) is guaranteed to be absent from the organic list and
// present only via injection.

const repoRoot = fileURLToPath(new URL("../..", import.meta.url));
const indexEntry = join(repoRoot, "src", "index.ts");
// Resolve the tsx binary itself (not `npx tsx`) so the child process needs
// no cwd-based lookup at all: npx would otherwise walk up from cwd looking
// for a local node_modules/.bin, which is exactly the coupling to repoRoot
// (and thus to any repo-root .env) this hermeticity fix removes.
const tsxBin = join(repoRoot, "node_modules", ".bin", "tsx");
// Every child process below runs with this as its cwd: the OS temp root
// always exists and never carries a project .env, so src/env.ts's
// `resolve(process.cwd(), ".env")` default can never silently refill a
// credential the test blanked. Do NOT use repoRoot or any of this file's
// own tmp dirs (they hold seeded fixtures, not env isolation) here.
const scratchCwd = tmpdir();

// Build the child env from scratch (an allowlist) rather than spreading
// `...process.env`: a denylist approach (deleting/blanking specific keys)
// silently reintroduces every future credential env var this suite doesn't
// yet know about, and inherited OPENAI_ADMIN_KEY / OPENAI_API_KEY et al.
// have previously caused an 86s+ hang against a real LLM endpoint. Only
// PATH/HOME/TMPDIR-family and the NODE_*/npm_config_* vars tsx itself may
// consult are carried over from the test runner's own environment; every
// ORACLE_* value is explicit per call site.
const INHERITED_ENV_KEYS = ["PATH", "HOME", "TMPDIR", "TEMP", "TMP"];
function buildChildEnv(
  extra: Record<string, string | undefined>,
): Record<string, string | undefined> {
  const env: Record<string, string | undefined> = {};
  for (const key of INHERITED_ENV_KEYS) {
    if (process.env[key] !== undefined) env[key] = process.env[key];
  }
  for (const key of Object.keys(process.env)) {
    if (key.startsWith("NODE_") || key.startsWith("npm_config_")) {
      env[key] = process.env[key];
    }
  }
  for (const [key, value] of Object.entries(extra)) {
    if (value === undefined) delete env[key];
    else env[key] = value;
  }
  return env;
}

const tmpDirs: string[] = [];
async function makeTmpDir(): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), "oracle-search-cli-"));
  tmpDirs.push(dir);
  return dir;
}

afterEach(async () => {
  while (tmpDirs.length > 0) {
    const dir = tmpDirs.pop();
    if (dir) await rm(dir, { recursive: true, force: true });
  }
});

async function makeRepo(
  scanRoot: string,
  name: string,
  files: Record<string, string>,
): Promise<void> {
  const repoDir = join(scanRoot, name);
  await mkdir(join(repoDir, ".git"), { recursive: true });
  for (const [rel, content] of Object.entries(files)) {
    const abs = join(repoDir, rel);
    await mkdir(join(abs, ".."), { recursive: true });
    await writeFile(abs, content, "utf8");
  }
}

// Every helper below spawns node_modules/.bin/tsx (absolute path) directly,
// never `npx tsx`: cwd is the hermetic scratchCwd (see above), and npx's own
// cwd-based node_modules lookup would defeat that. A generous timeout plus
// `killSignal` makes any future regression that resurrects a hanging real
// network call fail fast instead of hanging the whole suite.
const SPAWN_TIMEOUT_MS = 30_000;

function runIndex(
  scanRoot: string,
  dataDir: string,
): { stdout: string; stderr: string; status: number | null } {
  const result = spawnSync(
    tsxBin,
    [indexEntry, "index", "--path", scanRoot],
    {
      encoding: "utf8",
      cwd: scratchCwd,
      timeout: SPAWN_TIMEOUT_MS,
      killSignal: "SIGKILL",
      env: buildChildEnv({
        ORACLE_DATA_DIR: dataDir,
        ORACLE_EMBEDDING_PROVIDER: "stub",
        ORACLE_EMBEDDING_MODEL: "stub",
        ORACLE_SCAN_ROOT: scanRoot,
      }),
    },
  );
  return {
    stdout: result.stdout ?? "",
    stderr: result.stderr ?? "",
    status: result.status,
  };
}

function runSearch(
  dataDir: string,
  extraArgs: string[],
): { stdout: string; stderr: string; status: number | null } {
  const result = spawnSync(
    tsxBin,
    [
      indexEntry,
      "search",
      "design note for sources-expansion",
      "--path-glob",
      "**/design.md",
      "--limit",
      "5",
      ...extraArgs,
    ],
    {
      encoding: "utf8",
      cwd: scratchCwd,
      timeout: SPAWN_TIMEOUT_MS,
      killSignal: "SIGKILL",
      env: buildChildEnv({
        ORACLE_DATA_DIR: dataDir,
        ORACLE_EMBEDDING_PROVIDER: "stub",
        ORACLE_EMBEDDING_MODEL: "stub",
      }),
    },
  );
  return {
    stdout: result.stdout ?? "",
    stderr: result.stderr ?? "",
    status: result.status,
  };
}

function runCli(
  dataDir: string,
  args: string[],
): { stdout: string; stderr: string; status: number | null } {
  const result = spawnSync(tsxBin, [indexEntry, ...args], {
    encoding: "utf8",
    cwd: scratchCwd,
    timeout: SPAWN_TIMEOUT_MS,
    killSignal: "SIGKILL",
    env: buildChildEnv({
      ORACLE_DATA_DIR: dataDir,
      ORACLE_EMBEDDING_PROVIDER: "stub",
      ORACLE_EMBEDDING_MODEL: "stub",
    }),
  });
  return {
    stdout: result.stdout ?? "",
    stderr: result.stderr ?? "",
    status: result.status,
  };
}

function runCliWithEnv(
  dataDir: string,
  args: string[],
  extraEnv: Record<string, string | undefined>,
): { stdout: string; stderr: string; status: number | null } {
  const env = buildChildEnv({
    ORACLE_DATA_DIR: dataDir,
    ORACLE_EMBEDDING_PROVIDER: "stub",
    ORACLE_EMBEDDING_MODEL: "stub",
    // Belt-and-suspenders: the allowlist in buildChildEnv already excludes
    // every credential-shaped var by construction (it never carries over
    // anything outside INHERITED_ENV_KEYS / NODE_* / npm_config_*), but
    // pin these two explicitly so `auto` LLM-provider resolution can never
    // pick up a real key regardless of extraEnv's own choices below.
    ANTHROPIC_API_KEY: "",
    OPENAI_API_KEY: "",
    ...extraEnv,
  });
  const result = spawnSync(tsxBin, [indexEntry, ...args], {
    encoding: "utf8",
    cwd: scratchCwd,
    timeout: SPAWN_TIMEOUT_MS,
    killSignal: "SIGKILL",
    env,
  });
  return {
    stdout: result.stdout ?? "",
    stderr: result.stderr ?? "",
    status: result.status,
  };
}

describe("oracle search CLI sources-expansion integration", () => {
  it(
    "injects the [expanded from ...] marker by default; --no-expand-sources suppresses it",
    { timeout: 30_000 },
    async () => {
      const tmp = await makeTmpDir();
      const scanRoot = join(tmp, "repos");
      const dataDir = join(tmp, "data");
      await mkdir(scanRoot, { recursive: true });

      await makeRepo(scanRoot, "srcexp", {
        "docs/design.md": [
          "---",
          "type: doc",
          "sources:",
          "  - srcexp/src/impl.ts",
          "---",
          "",
          "# Design note",
          "",
          "This design note exists purely to exercise sources-expansion in an integration test.",
        ].join("\n"),
        "src/impl.ts":
          'export function implementedThing(): string {\n  return "impl";\n}\n',
      });

      const indexResult = runIndex(scanRoot, dataDir);
      expect(indexResult.status, `index failed: ${indexResult.stderr}`).toBe(
        0,
      );

      // Default: expansion on. The path-glob pins the organic list to the
      // design doc alone, so src/impl.ts's chunk body can only appear via
      // injection. Note: the parent's own rendered `sources: srcexp/src/impl.ts`
      // line ALSO contains the string "src/impl.ts" regardless of expansion,
      // so the assertions below key on the injected chunk's unique page
      // content ("implementedThing") and the injected row's own header,
      // rather than the bare path substring.
      const withExpansion = runSearch(dataDir, []);
      expect(
        withExpansion.status,
        `search failed: ${withExpansion.stderr}`,
      ).toBe(0);
      expect(withExpansion.stdout).toContain("docs/design.md");
      expect(withExpansion.stdout).toContain("[expanded from design.md]");
      expect(withExpansion.stdout).toContain("--- srcexp/src/impl.ts");
      expect(withExpansion.stdout).toContain("implementedThing");

      // --no-expand-sources: injection suppressed entirely. The parent's own
      // `sources: ...` line still renders (unrelated to expansion), but the
      // injected chunk's page content and header must be absent.
      const withoutExpansion = runSearch(dataDir, ["--no-expand-sources"]);
      expect(
        withoutExpansion.status,
        `search failed: ${withoutExpansion.stderr}`,
      ).toBe(0);
      expect(withoutExpansion.stdout).toContain("docs/design.md");
      expect(withoutExpansion.stdout).not.toContain("[expanded from");
      expect(withoutExpansion.stdout).not.toContain("--- srcexp/src/impl.ts");
      expect(withoutExpansion.stdout).not.toContain("implementedThing");
    },
  );

  it("emits complete, machine-readable JSON for search, list, and expand", { timeout: 30_000 }, async () => {
    const tmp = await makeTmpDir();
    const scanRoot = join(tmp, "repos");
    const dataDir = join(tmp, "data");
    await mkdir(scanRoot, { recursive: true });
    const longText = `marker-${"x".repeat(620)}`;
    await makeRepo(scanRoot, "jsonrepo", {
      "docs/long.md": longText,
    });
    expect(runIndex(scanRoot, dataDir).status).toBe(0);

    const search = runCli(dataDir, [
      "search", "marker", "--repo", "jsonrepo", "-k", "3", "--json",
    ]);
    expect(search.status, search.stderr).toBe(0);
    expect(search.stdout.startsWith("{")).toBe(true);
    expect(search.stdout).not.toContain("Loaded ");
    const searchJson = JSON.parse(search.stdout);
    expect(searchJson).toMatchObject({ ok: true, query: "marker", repo: "jsonrepo", limit: 3 });
    expect(searchJson.results.length).toBeLessThanOrEqual(3);
    expect(searchJson.results[0]).toEqual(expect.objectContaining({
      repo: "jsonrepo",
      filePath: "jsonrepo/docs/long.md",
      lineStart: expect.any(Number),
      lineEnd: expect.any(Number),
      fmType: null,
      fmTags: null,
      fmSources: null,
      expandedFrom: null,
      text: expect.stringContaining("marker-"),
    }));
    expect(searchJson.results[0].text.length).toBeGreaterThan(500);

    const list = runCli(dataDir, ["list-repos", "--json"]);
    expect(list.status, list.stderr).toBe(0);
    const listJson = JSON.parse(list.stdout);
    expect(listJson.ok).toBe(true);
    expect(listJson.repos[0]).toEqual(expect.objectContaining({
      repo: "jsonrepo", chunkCount: expect.any(Number), fileCount: 1,
      lastIndexedAt: expect.any(String), skippedSizeCount: 0,
      skippedErrorCount: 0, skippedExamples: [],
    }));

    const emptyList = runCli(join(tmp, "empty-data"), ["list-repos", "--json"]);
    expect(emptyList.status, emptyList.stderr).toBe(0);
    expect(JSON.parse(emptyList.stdout)).toEqual({ ok: true, repos: [] });

    const expand = runCli(dataDir, [
      "expand", "jsonrepo", "jsonrepo/docs/long.md", "--json",
    ]);
    expect(expand.status, expand.stderr).toBe(0);
    expect(JSON.parse(expand.stdout)).toEqual(expect.objectContaining({
      ok: true, repo: "jsonrepo", path: "jsonrepo/docs/long.md",
      lineStart: 1, lineEnd: expect.any(Number), totalLines: expect.any(Number),
      text: expect.stringContaining("marker-"),
    }));

    const missing = runCli(dataDir, [
      "expand", "jsonrepo", "missing.ts", "--json",
    ]);
    expect(missing.status).not.toBe(0);
    expect(JSON.parse(missing.stdout)).toEqual(expect.objectContaining({
      ok: false, reason: "not_indexed", message: expect.any(String),
    }));
  });

  it(
    "query --json marks a normal answer ok: true with no degraded key",
    { timeout: 30_000 },
    async () => {
      const tmp = await makeTmpDir();
      const scanRoot = join(tmp, "repos");
      const dataDir = join(tmp, "data");
      await mkdir(scanRoot, { recursive: true });
      await makeRepo(scanRoot, "queryrepo", {
        "src/thing.ts": "export function thing() { return 1; }\n",
      });
      expect(runIndex(scanRoot, dataDir).status).toBe(0);

      // auto + no credentials -> createLlm returns null -> raw-context
      // answer, NOT the LLM-failure branch: ok: true, no degraded key.
      const result = runCliWithEnv(
        dataDir,
        ["query", "what does thing do?", "--json"],
        { ORACLE_LLM_PROVIDER: "auto" },
      );
      expect(result.status, result.stderr).toBe(0);
      const doc = JSON.parse(result.stdout);
      expect(doc.ok).toBe(true);
      expect(doc).not.toHaveProperty("degraded");
      expect(doc).not.toHaveProperty("degradedReason");
    },
  );

  // The LLM-failure -> degraded fallback is also exercised at the unit
  // level (tests/unit/query-codebase.test.ts, "LLM invoke-failure branch")
  // via the deps.createLlm injection seam, and formatQueryJson's rendering
  // of degraded/degradedReason in tests/unit/format-json.test.ts.
  it(
    "query --json marks an LLM-failure fallback degraded: true at the CLI level",
    { timeout: 30_000 },
    async () => {
      const tmp = await makeTmpDir();
      const scanRoot = join(tmp, "repos");
      const dataDir = join(tmp, "data");
      await mkdir(scanRoot, { recursive: true });
      await makeRepo(scanRoot, "degradedrepo", {
        "src/thing.ts": "export function thing() { return 1; }\n",
      });
      expect(runIndex(scanRoot, dataDir).status).toBe(0);

      // ORACLE_LLM_PROVIDER=openai-compatible with an empty API key makes
      // the OpenAI SDK's client construction throw "Missing credentials"
      // synchronously on invoke, before any network I/O: no real request
      // is attempted, so this is instant and safe (unlike a genuinely
      // closed port, which the LLM constructors do not currently time out
      // quickly against). ORACLE_LLM_BASE_URL is still set to a closed
      // local port so the run stays fully offline even if that behavior
      // ever changes upstream.
      const result = runCliWithEnv(
        dataDir,
        ["query", "what does thing do?", "--json"],
        {
          ORACLE_LLM_PROVIDER: "openai-compatible",
          ORACLE_LLM_BASE_URL: "http://127.0.0.1:9/v1",
          ORACLE_LLM_API_KEY: "",
        },
      );
      expect(result.status, result.stderr).toBe(0);
      const doc = JSON.parse(result.stdout);
      expect(doc.ok).toBe(true);
      expect(doc.degraded).toBe(true);
      expect(doc.degradedReason).toBe("llm_request_failed");
    },
  );

  it(
    "stays hermetic against real credentials inherited from the parent process env",
    { timeout: 20_000 },
    async () => {
      const tmp = await makeTmpDir();
      const scanRoot = join(tmp, "repos");
      const dataDir = join(tmp, "data");
      await mkdir(scanRoot, { recursive: true });
      await makeRepo(scanRoot, "hermeticrepo", {
        "src/thing.ts": "export function thing() { return 1; }\n",
      });
      expect(runIndex(scanRoot, dataDir).status).toBe(0);

      // Simulate the test runner's own process having inherited real
      // credentials (a shell profile export, or a repo-root .env a
      // developer has locally). buildChildEnv's allowlist must exclude
      // these by construction, regardless of what the parent process
      // carries: this is the proof for that, not another case-by-case
      // blank. A leaked key would let the SDK attempt a real request
      // against the closed local port instead of failing synchronously on
      // missing credentials, which previously hung the suite for 86s+.
      const prevAdminKey = process.env.OPENAI_ADMIN_KEY;
      const prevApiKey = process.env.OPENAI_API_KEY;
      process.env.OPENAI_ADMIN_KEY = "sk-admin-should-never-reach-child";
      process.env.OPENAI_API_KEY = "sk-should-never-reach-child";
      try {
        const start = Date.now();
        const result = runCliWithEnv(
          dataDir,
          ["query", "what does thing do?", "--json"],
          {
            ORACLE_LLM_PROVIDER: "openai-compatible",
            ORACLE_LLM_BASE_URL: "http://127.0.0.1:9/v1",
            ORACLE_LLM_API_KEY: "",
          },
        );
        const elapsedMs = Date.now() - start;
        expect(result.status, result.stderr).toBe(0);
        const doc = JSON.parse(result.stdout);
        expect(doc.ok).toBe(true);
        expect(doc.degraded).toBe(true);
        expect(doc.degradedReason).toBe("llm_request_failed");
        // Bound well under SPAWN_TIMEOUT_MS: a leaked credential reaching
        // the child is the failure mode this test exists to catch, and it
        // shows up as a hang against the closed port, not as a wrong
        // JSON value.
        expect(elapsedMs).toBeLessThan(15_000);
      } finally {
        if (prevAdminKey === undefined) delete process.env.OPENAI_ADMIN_KEY;
        else process.env.OPENAI_ADMIN_KEY = prevAdminKey;
        if (prevApiKey === undefined) delete process.env.OPENAI_API_KEY;
        else process.env.OPENAI_API_KEY = prevApiKey;
      }
    },
  );

  it("returns one JSON error document for pre-action errors on JSON-capable commands", { timeout: 20_000 }, () => {
    for (const args of [
      ["query", "--json"],
      ["search", "term", "--json", "--unknown"],
      ["expand", "repo", "--json"],
      ["list-repos", "--json", "--unknown"],
    ]) {
      const result = runCli(join(tmpdir(), "unused-oracle-json-errors"), args);
      expect(result.status, `${args.join(" ")}: ${result.stderr}`).not.toBe(0);
      expect(result.stdout.startsWith("{")).toBe(true);
      expect(result.stdout.trim().split("\n")).toHaveLength(1);
      expect(JSON.parse(result.stdout)).toEqual({
        ok: false,
        error: { message: expect.any(String) },
      });
    }

    const textError = runCli(join(tmpdir(), "unused-oracle-text-error"), ["query"]);
    expect(textError.status).not.toBe(0);
    expect(textError.stdout).toBe("");
    expect(textError.stderr).toContain("missing required argument 'question'");

    const globalJson = runCli(join(tmpdir(), "unused-oracle-global-json"), [
      "--json", "list-repos",
    ]);
    expect(globalJson.status).not.toBe(0);
    expect(globalJson.stdout).toBe("");
    expect(globalJson.stderr).toContain("unknown option '--json'");
  });
});
