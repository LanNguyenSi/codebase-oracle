# cli-toy

Minimal commander-based CLI used as eval fixture for codebase-oracle.

The entry point in `src/index.ts` wires two subcommands:

- `cli-toy list` lists items.
- `cli-toy create <name>` creates one.

Each subcommand body lives under `src/commands/`. The eval corpus uses
this shape to exercise oracle's retrieval on the kind of "where is the
CLI argument parsing for subcommand X" question that comes up
constantly in real repos.
