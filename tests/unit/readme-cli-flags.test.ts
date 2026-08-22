import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import type { Option } from "commander";
import { buildProgram } from "../../src/index.js";

// Guards the README's "## CLI reference" flag table against drift from the
// commander option registrations in src/index.ts: the 2026-08-21 audit found
// `-g, --path-glob` missing from the README table (PR #84) and a stale `-k`
// default annotation (PR #83). This test fails with a readable diff if a
// flag is added to commander without a matching README row, or vice versa,
// and checks that a row documenting a default value states the actual
// commander default(s).

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "../..");
const readme = readFileSync(resolve(repoRoot, "README.md"), "utf8");

interface ReadmeRow {
  /** Long flag name without leading dashes, e.g. "repo". */
  flag: string;
  raw: string;
}

function parseReadmeFlagTable(markdown: string): ReadmeRow[] {
  const section = markdown.match(
    /## CLI reference[\s\S]*?\| Flag \| Description \|\n\|[-\s|]+\|\n([\s\S]*?)\n\n/,
  );
  if (!section) {
    throw new Error(
      "Could not locate the '## CLI reference' flag table in README.md",
    );
  }
  const rows: ReadmeRow[] = [];
  for (const line of section[1].split("\n")) {
    if (!line.trim().startsWith("|")) continue;
    const longFlagMatch = line.match(/--([a-zA-Z][a-zA-Z0-9-]*)/);
    if (!longFlagMatch) continue;
    rows.push({ flag: longFlagMatch[1], raw: line });
  }
  return rows;
}

interface CommanderFlag {
  command: string;
  flag: string;
  defaultValue: unknown;
}

function collectCommanderFlags(commandNames: string[]): CommanderFlag[] {
  const program = buildProgram();
  const flags: CommanderFlag[] = [];
  for (const name of commandNames) {
    const command = program.commands.find((c) => c.name() === name);
    if (!command) {
      throw new Error(`Expected commander to register a "${name}" command`);
    }
    for (const option of command.options as Option[]) {
      flags.push({
        command: name,
        flag: option.long ? option.long.replace(/^--/, "") : option.name(),
        defaultValue: option.defaultValue,
      });
    }
  }
  return flags;
}

describe("README CLI flag table vs commander registrations", () => {
  // The README's flag table (below the `## CLI reference` usage block) only
  // documents the `query`/`search` commands' options; `index`/`expand`/`watch`
  // are documented via the usage examples above the table instead.
  const commanderFlags = collectCommanderFlags(["query", "search"]);
  const readmeRows = parseReadmeFlagTable(readme);

  it("has a README row for every query/search commander option", () => {
    const commanderFlagNames = [...new Set(commanderFlags.map((f) => f.flag))];
    const readmeFlagNames = new Set(readmeRows.map((r) => r.flag));
    const missing = commanderFlagNames.filter((f) => !readmeFlagNames.has(f));
    expect(
      missing,
      `Flags registered in src/index.ts (query/search) but missing from the README ` +
        `"## CLI reference" table: ${missing.join(", ") || "(none)"}`,
    ).toEqual([]);
  });

  it("has no README row for a flag that is not registered on query/search", () => {
    const commanderFlagNames = new Set(commanderFlags.map((f) => f.flag));
    const extra = readmeRows
      .map((r) => r.flag)
      .filter((f) => !commanderFlagNames.has(f));
    expect(
      extra,
      `README "## CLI reference" table rows referencing flags not registered on the ` +
        `query/search commands in src/index.ts: ${extra.join(", ") || "(none)"}`,
    ).toEqual([]);
  });

  it("documents each commander default value in its README row", () => {
    const byFlag = new Map<string, CommanderFlag[]>();
    for (const f of commanderFlags) {
      if (f.defaultValue === undefined) continue;
      const list = byFlag.get(f.flag) ?? [];
      list.push(f);
      byFlag.set(f.flag, list);
    }

    const mismatches: string[] = [];
    for (const [flag, entries] of byFlag) {
      const row = readmeRows.find((r) => r.flag === flag);
      if (!row) continue; // already reported by the "missing row" check above
      for (const entry of entries) {
        const expected = String(entry.defaultValue);
        if (!row.raw.includes(expected)) {
          mismatches.push(
            `--${flag}: README row does not mention the "${entry.command}" ` +
              `command's default (${expected}). Row: ${row.raw.trim()}`,
          );
        }
      }
    }
    expect(mismatches, mismatches.join("\n")).toEqual([]);
  });
});
