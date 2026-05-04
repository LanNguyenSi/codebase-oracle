# Contributing to codebase-oracle

Thanks for your interest. codebase-oracle is a local-first MCP server that semantically indexes your repos.

## Issues

- Bug reports: include repro steps, expected vs. actual, the verb (`oracle_search`, `oracle_query`, `oracle_expand`, `oracle_list_repos`), Node version, and OS.
- Feature requests: describe the use case before the proposed shape.

## Pull Requests

1. Fork, branch off `master` (e.g. `feat/<scope>`, `fix/<scope>`).
2. Keep changes scoped where possible.
3. Run the local checks:

   ```bash
   npm install
   npm run build
   npm test
   ```

4. After native-dep changes (`better-sqlite3`, `sqlite-vec`), verify the install boundary loads cleanly: `node -e "require('better-sqlite3'); require('sqlite-vec')"`.
5. Open the PR with a clear summary, motivation, and test plan.

## Dev Setup

```bash
git clone https://github.com/LanNguyenSi/codebase-oracle.git
cd codebase-oracle
npm install
npm run build
```

Register as a Claude Code MCP server per `README.md` once `dist/` is built.

## Style

Match the surrounding code. Prefer small, reviewable diffs.
