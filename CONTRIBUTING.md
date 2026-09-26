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

## Releasing

Retrieval quality is guarded by a hand-labelled eval set rather than by CI. The
eval needs an embedding provider (`OPENAI_API_KEY`, or an OpenAI-compatible
endpoint such as Ollama) and costs under a cent per run, so it runs as a
**manual pre-release gate**, not on every PR:

```bash
npm run eval           # compares retrieval against tests/eval/baseline.json
```

Run it before tagging a release and paste the final line into the release PR. A
regression vs. baseline blocks the release until the cause is fixed or the
baseline is updated with a documented reason. See
[tests/eval/README.md](tests/eval/README.md) for the full workflow, including
how to add questions and corpus repos.

Two guards keep the tag and the changelog from drifting apart: a vitest unit
test fails when `CHANGELOG.md`'s first `## [x.y.z] - date` heading doesn't
match `package.json`'s version, and the release workflow's changelog-extraction
step (`scripts/extract-changelog-notes.sh`) fails loudly if no release notes
are found for the tagged version.
