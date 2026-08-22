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
// flag is added to commander without a matching README row, if a row is
// scoped to the wrong command (e.g. moved from `search` to `query` in
// commander but the README still says "(`search` only)"), or if a row's
// documented default value stops matching the actual commander default.

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "../..");
const readme = readFileSync(resolve(repoRoot, "README.md"), "utf8");

// The README's flag table (below the `## CLI reference` usage block) only
// documents the `query`/`search` commands' options; `index`/`expand`/`watch`
// are documented via the usage examples above the table instead.
const READABLE_COMMANDS = ["query", "search"] as const;

// The full set of commands buildProgram() is expected to register. This is
// a conscious allowlist, not derived from the program itself: if a new
// command is added, the test below fails until this list (and, if the new
// command's flags belong in the README table, READABLE_COMMANDS) is updated
// by hand, rather than silently leaving the new command's flags unchecked.
const ALL_KNOWN_COMMANDS = [
  "mcp",
  "index",
  "query",
  "search",
  "list-repos",
  "expand",
  "migrate-store",
  "watch",
];

interface ReadmeRow {
  /** Long flag name without leading dashes, e.g. "repo". */
  flag: string;
  raw: string;
  /** Commands this row documents (parsed from a "(`<cmd>` only)" prefix; a
   *  row without that prefix is assumed to document every readable command
   *  that registers the flag). */
  commands: string[];
  /** Command name -> documented default value text, parsed from a
   *  "(default: ...)" clause in the row's description. */
  defaultByCommand: Map<string, string>;
}

const COMMAND_ONLY_RE = /\(`(query|search)`\s+only\)/;

function parseDocumentedDefaults(
  raw: string,
  commands: string[],
): Map<string, string> {
  const result = new Map<string, string>();
  const clauseMatch = raw.match(/default:\s*(.+?)\)/);
  if (!clauseMatch) return result;
  const clause = clauseMatch[1];
  for (const command of commands) {
    // Per-command breakdown, e.g. "12 for `query`, 10 for `search`".
    const perCommand = clause.match(
      new RegExp(`(\\S+)\\s+for\\s+\`${command}\``),
    );
    if (perCommand) {
      result.set(command, perCommand[1]);
    } else if (commands.length === 1) {
      // Only one command documents this flag, so the whole clause is its
      // default (no "for `command`" breakdown needed).
      result.set(command, clause.trim());
    }
  }
  return result;
}

function parseReadmeFlagTable(
  markdown: string,
  knownCommands: readonly string[],
): ReadmeRow[] {
  const section = markdown.match(
    /## CLI reference[\s\S]*?\| Flag \| Description \|\n\|[-\s|]+\|\n([\s\S]*?)(?:\n\n|$)/,
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
    const flag = longFlagMatch[1];
    const restriction = line.match(COMMAND_ONLY_RE);
    const commands = restriction ? [restriction[1]] : [...knownCommands];
    rows.push({
      flag,
      raw: line,
      commands,
      defaultByCommand: parseDocumentedDefaults(line, commands),
    });
  }
  return rows;
}

interface CommanderFlag {
  command: string;
  flag: string;
  defaultValue: unknown;
}

function collectCommanderFlags(
  commandNames: readonly string[],
): CommanderFlag[] {
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
  it("does not register a command outside the ALL_KNOWN_COMMANDS allowlist", () => {
    const program = buildProgram();
    const actual = program.commands.map((c) => c.name()).sort();
    expect(
      actual,
      "buildProgram() registered a command not listed in this test's " +
        "ALL_KNOWN_COMMANDS. If this is a real new command, add it to " +
        "ALL_KNOWN_COMMANDS and decide whether it belongs in " +
        "READABLE_COMMANDS (and the README table) too.",
    ).toEqual([...ALL_KNOWN_COMMANDS].sort());
  });

  const commanderFlags = collectCommanderFlags(READABLE_COMMANDS);
  const readmeRows = parseReadmeFlagTable(readme, READABLE_COMMANDS);
  const commanderPairs = new Set(
    commanderFlags.map((f) => `${f.command}:${f.flag}`),
  );
  const readmePairs = new Set<string>();
  for (const row of readmeRows) {
    for (const command of row.commands) readmePairs.add(`${command}:${row.flag}`);
  }

  it("has a README row for every query/search commander option, scoped to the right command", () => {
    const missing = [...commanderPairs].filter((p) => !readmePairs.has(p));
    expect(
      missing,
      `command:flag pairs registered in src/index.ts (query/search) but missing (or wrongly ` +
        `scoped) in the README "## CLI reference" table: ${missing.join(", ") || "(none)"}`,
    ).toEqual([]);
  });

  it("has no README row documenting a command:flag pair that is not registered", () => {
    const extra = [...readmePairs].filter((p) => !commanderPairs.has(p));
    expect(
      extra,
      `README "## CLI reference" table rows documenting a command:flag pair not registered on ` +
        `the query/search commands in src/index.ts: ${extra.join(", ") || "(none)"}`,
    ).toEqual([]);
  });

  it("documents each commander default value for the correct command in its README row", () => {
    const rowByFlag = new Map(readmeRows.map((r) => [r.flag, r]));
    const mismatches: string[] = [];
    for (const entry of commanderFlags) {
      if (entry.defaultValue === undefined) continue;
      const row = rowByFlag.get(entry.flag);
      if (!row) continue; // already reported by the "missing row" check above
      const expected = String(entry.defaultValue);
      const documented = row.defaultByCommand.get(entry.command);
      if (documented !== expected) {
        mismatches.push(
          `--${entry.flag}: README row does not document "${expected}" as the "${entry.command}" ` +
            `command's default (found ${documented === undefined ? "nothing" : `"${documented}"`} instead). ` +
            `Row: ${row.raw.trim()}`,
        );
      }
    }
    expect(mismatches, mismatches.join("\n")).toEqual([]);
  });
});
