#!/usr/bin/env node
// cli-toy entry point. Defines the commander program and dispatches to
// per-subcommand handlers under ./commands/.

import { Command } from "commander";
import { listItems } from "./commands/list.js";
import { createItem } from "./commands/create.js";

const program = new Command();

program
  .name("cli-toy")
  .description("Tiny example CLI used as a codebase-oracle eval fixture.")
  .version("0.1.0");

program
  .command("list")
  .description("List all items.")
  .option("-l, --limit <n>", "Limit the number of items printed", "10")
  .action((opts) => {
    listItems({ limit: parseInt(opts.limit, 10) });
  });

program
  .command("create")
  .description("Create a new item.")
  .argument("<name>", "Item name")
  .option("-t, --tag <tag>", "Optional tag")
  .action((name: string, opts) => {
    createItem({ name, tag: opts.tag });
  });

program.parse();
