# Smoke-Test 2026-05-11: oracle search evaluation

Manual end-to-end check of `oracle_list_repos`, `oracle_search`, and
`oracle_query` against the production index. Recorded as the paper trail
for agent-tasks task
[`501e96bb-0157-4449-b189-37449742338a`](https://agent-tasks.opentriologue.ai/tasks/501e96bb-0157-4449-b189-37449742338a).

## Scope

Four scenarios from the task description:

1. Repo listing precondition (`oracle_list_repos` non-empty).
2. Semantic search vs. symbol-name query (recall, top hits, false positives).
3. Recency / reindex latency.
4. Cross-repo scoring on a shared script reference.

## Result summary

| Aspect                 | Status | Note                                                |
|------------------------|--------|-----------------------------------------------------|
| Repo listing           | ok     | 39 repos, dist healthy                              |
| Semantic recall        | warn   | Archived docs outrank active code on auth queries   |
| Symbol recall          | fail   | `oracle_query` 500 + vendor-cache skews ranking     |
| Index freshness        | fail   | 6 days stale, no scheduled reindex                  |
| Cross-repo file lookup | warn   | Concept-level OK, filename fan-out missing          |

Full scenario logs (queries, top hits, raw output) live in the task
comment trail on agent-tasks, not duplicated here.

## Follow-ups filed

- `46d0d710-7f11-4ab5-a2c3-1804779b5bf2` (HIGH): exclude vendor caches
  and `node_modules` from indexing.
- `d96e47dc-3cc3-4801-acf1-f54946dcb955` (MEDIUM): schedule automatic
  reindex (cron + optional MCP-triggered).

## Repros confirmed

- Task `7549a1ce-0139-4cd1-b85d-5e94dfca1968` (oracle_query 500):
  reproduced with request id `e0994fc6-2fdc-4e31-8d15-64329a100afa` on
  a symbol-name question.
