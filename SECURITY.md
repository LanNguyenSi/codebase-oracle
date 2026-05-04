# Security Policy

## Supported Versions

Active development is on `master`. Only the latest tagged release is patched.

codebase-oracle indexes local source files and ships with native deps (`better-sqlite3`, `sqlite-vec`). Vulnerabilities involving path traversal, indexing of sensitive files, or SQLite-side issues are treated as serious.

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security reports.

Email **contact@lan-nguyen-si.de** with:

- Affected version
- Reproduction steps or proof-of-concept
- Impact assessment

You will get an acknowledgement within 72 hours and an initial assessment within 7 days. A fix timeline depends on severity and complexity, communicated in the assessment.
