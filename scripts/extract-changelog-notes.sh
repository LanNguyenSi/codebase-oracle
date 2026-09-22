#!/usr/bin/env bash
# Extracts the release notes body for one version from a Keep-a-Changelog
# style CHANGELOG.md: everything between "## [<version>] ..." and the next
# "## [" heading (or end of file). Used by .github/workflows/release.yml and
# replayed directly by tests/unit/changelog-extraction.test.ts, so the
# extraction logic lives in exactly one place.
#
# The version is passed as a plain positional argument (an awk -v variable,
# never interpolated into the awk program text itself), so a tag-derived
# version containing awk/regex metacharacters cannot change what the program
# matches.
#
# Usage: extract-changelog-notes.sh <version> [changelog-path]
# Exit 0 with the notes on stdout when the heading exists and its body has
# non-whitespace content; exit 1 with an explanatory message on stderr
# otherwise.
set -euo pipefail

version="${1:?usage: extract-changelog-notes.sh <version> [changelog-path]}"
changelog="${2:-CHANGELOG.md}"

if [ ! -f "$changelog" ]; then
  echo "::error::changelog file not found: $changelog" >&2
  exit 1
fi

notes=$(awk -v ver="$version" '
  index($0, "## [" ver "]") == 1 { found=1; next }
  /^## \[/ { found=0 }
  found
' "$changelog")

if ! printf '%s' "$notes" | grep -q '[^[:space:]]'; then
  echo "::error::no changelog notes extracted for version $version (missing a '## [$version] - date' heading in $changelog?)" >&2
  exit 1
fi

printf '%s\n' "$notes"
