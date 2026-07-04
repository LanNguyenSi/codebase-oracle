---
type: doc
sources:
  - cli-toy/src/commands/create.ts
---

# Why `create` accepts an optional label

Product wants every newly created item to be easy to group later, so the
`create` subcommand accepts an optional short label that gets attached to
the item at creation time. When a caller omits the label, the item is
created bare with no grouping information; when they supply one, it is
appended to the confirmation line so the operator can see at a glance
which group the new item joined. The label is intentionally free-form
(no enum, no validation) because early users kept inventing ad-hoc
categories faster than we could enumerate them.
