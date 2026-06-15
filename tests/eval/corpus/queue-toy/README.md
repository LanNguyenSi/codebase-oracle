# queue-toy

Minimal BullMQ-style job queue used as a codebase-oracle eval fixture.

- `src/queue.ts` - queue and scheduler setup, connection config, job options
- `src/worker.ts` - worker instantiation and per-job-type handler dispatch
