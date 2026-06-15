# Retrieval-quality eval set

What lives here is the regression net for "does codebase-oracle still return the right chunks for the kind of question users actually ask?". Unit tests cover the chunker, splitter, store, and embedding fingerprint, but none of them tell us whether retrieval quality drifted after a change to the embedding model, the chunking strategy, or the scoring formula. This is the missing piece.

## Layout

```
tests/eval/
├── README.md                  // you are here
├── runner.ts                  // npm run eval
├── questions.json             // hand-labelled Q&A pairs
├── baseline.json              // last known good per-question pass/fail
└── corpus/                    // small public/sample code, walked at run time
    ├── .codebase-oracle-skip  // sentinel; the main scanner prunes this tree
    ├── auth-toy/              // toy JWT auth surface
    ├── cli-toy/               // toy commander CLI
    ├── config-toy/            // toy zod-based env validation
    ├── server-toy/            // toy Fastify server (app bootstrap + route defs)
    ├── queue-toy/             // toy BullMQ job runner (queue setup + worker)
    ├── form-toy/              // toy React contact form + validation hook
    └── db-toy/                // toy Prisma DB layer (schema + query helpers)
```

The `.codebase-oracle-skip` sentinel at `corpus/` is what stops the user's primary index from absorbing fixture content when it walks this repo. The eval runner sets its `scanRoot` to `corpus/` and walks each per-repo subdir, so it never trips on the sentinel itself.

## Running

```bash
# Real embeddings (OPENAI_API_KEY by default; Ollama via the
# openai-compatible provider is fine too). Compares against baseline.json
# and exits non-zero on any regression.
npm run eval

# Same run, but overwrite baseline.json with the current results.
# Use this only after an intentional improvement (eg new embedding
# model that bumps a question to a better rank) and review the diff.
npm run eval -- --update
```

Cost is small: ~25 documents to embed for a full corpus rebuild, plus one embed per question. Under a cent against `text-embedding-3-small`.

## Adding a question

1. Append a record to `questions.json`:

   ```json
   {
     "id": "short-stable-slug",
     "question": "what you would actually type into the oracle",
     "expectedFilePaths": [
       { "path": "auth-toy/src/token.ts", "atRank": "top-3" }
     ]
   }
   ```

   `id` is what regression reports cite; keep it short and stable. `atRank` accepts any `top-N` bucket.

2. Run `npm run eval` once to confirm the new question already passes against the current corpus.
3. Run `npm run eval -- --update` to bake it into `baseline.json`.

Avoid leaking real proprietary content into the corpus. Add or extend a toy repo instead so the fixture stays public.

## Adding a corpus repo

Each fixture repo lives at `corpus/<name>/` and is treated as a real repo by the indexer because the eval runner materialises a `.git` placeholder file in each subdir before indexing. Git refuses to track paths literally named `.git`, so the placeholders are local-only (gitignored) and re-created on every `npm run eval`. To add a new fixture repo, create `corpus/<name>/` with a `README.md` plus source files, commit those, and the next `npm run eval` will mint the `.git` marker for you.

## Pre-release hook

Following the `feedback_release_dogfood` pattern: before tagging a release, run `npm run eval` and paste the final line into the release PR. A regression vs. baseline blocks merge until either the cause is fixed or the baseline is updated with a documented reason.
