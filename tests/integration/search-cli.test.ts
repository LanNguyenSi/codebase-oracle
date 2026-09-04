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

function runIndex(
  scanRoot: string,
  dataDir: string,
): { stdout: string; stderr: string; status: number | null } {
  const result = spawnSync(
    "npx",
    ["tsx", indexEntry, "index", "--path", scanRoot],
    {
      encoding: "utf8",
      cwd: repoRoot,
      env: {
        ...process.env,
        ORACLE_DATA_DIR: dataDir,
        ORACLE_EMBEDDING_PROVIDER: "stub",
        ORACLE_EMBEDDING_MODEL: "stub",
        ORACLE_SCAN_ROOT: scanRoot,
      },
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
    "npx",
    [
      "tsx",
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
      cwd: repoRoot,
      env: {
        ...process.env,
        ORACLE_DATA_DIR: dataDir,
        ORACLE_EMBEDDING_PROVIDER: "stub",
        ORACLE_EMBEDDING_MODEL: "stub",
      },
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
  const result = spawnSync("npx", ["tsx", indexEntry, ...args], {
    encoding: "utf8",
    cwd: repoRoot,
    env: {
      ...process.env,
      ORACLE_DATA_DIR: dataDir,
      ORACLE_EMBEDDING_PROVIDER: "stub",
      ORACLE_EMBEDDING_MODEL: "stub",
    },
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
    expect(searchJson).toMatchObject({ query: "marker", repo: "jsonrepo", limit: 3 });
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
    expect(JSON.parse(list.stdout).repos[0]).toEqual(expect.objectContaining({
      repo: "jsonrepo", chunkCount: expect.any(Number), fileCount: 1,
      lastIndexedAt: expect.any(String), skippedSizeCount: 0,
      skippedErrorCount: 0, skippedExamples: [],
    }));

    const emptyList = runCli(join(tmp, "empty-data"), ["list-repos", "--json"]);
    expect(emptyList.status, emptyList.stderr).toBe(0);
    expect(JSON.parse(emptyList.stdout)).toEqual({ repos: [] });

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

  it("returns one JSON error document for pre-action errors on JSON-capable commands", () => {
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
