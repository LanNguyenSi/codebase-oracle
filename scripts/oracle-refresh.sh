#!/usr/bin/env bash
# codebase-oracle: scheduled refresh for a machine that serves as the index
# source of truth.
#
# Fast-forwards every clean, non-detached checkout with an upstream that sits
# directly under ORACLE_SCAN_ROOT, then runs the incremental index. Meant to
# be driven by a scheduler (systemd timer, launchd job, cron) so the index
# host stays current without a human running `npm run index` by hand.
#
# A dirty checkout, a detached HEAD, an unreadable checkout, or a branch with
# no upstream is skipped rather than touched; a failed pull is reported (with
# git's stderr reason) and the loop continues so one bad repo doesn't block
# the rest. Hidden directories directly under ORACLE_SCAN_ROOT are not seen
# by the pull loop (the glob below skips dotfiles the same way the
# indexer's own discoverRepos does in src/ingest/scanner.ts), so parity with
# what actually gets indexed holds. The index step always runs afterward
# (unless ORACLE_REFRESH_PULL=0 is set to skip the pull phase entirely), and
# its exit status is this script's exit status.
#
# Env knobs:
#   ORACLE_SCAN_ROOT        Directory whose direct child repos get pulled.
#                           Falls back to the ORACLE_SCAN_ROOT= line in the
#                           checkout's .env when unset.
#   ORACLE_REFRESH_PULL     Set to 0 to skip the pull phase and only index.
#   ORACLE_REFRESH_INDEX_CMD  Command to run for the index step, executed via
#                           `bash -c` in the checkout directory. Defaults to
#                           `npm run index`. This is operator-controlled and
#                           executed as a shell command; treat it as the same
#                           trust tier as the plist or unit file that invokes
#                           this script.
#   ORACLE_REFRESH_ENV_FILE  Testing aid only: overrides the .env path used
#                           by the ORACLE_SCAN_ROOT fallback below (default
#                           "<checkout>/.env").
set -u

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
checkout_dir="$(cd -- "${script_dir}/.." >/dev/null 2>&1 && pwd -P)"
cd -- "${checkout_dir}" || exit 1

resolve_scan_root() {
  if [ -n "${ORACLE_SCAN_ROOT:-}" ]; then
    printf '%s\n' "${ORACLE_SCAN_ROOT}"
    return 0
  fi

  local env_file="${ORACLE_REFRESH_ENV_FILE:-${checkout_dir}/.env}"
  if [ -f "${env_file}" ]; then
    local line value
    # Match src/env.ts's own .env parser: allow whitespace around the key
    # and the "=" (grep -m1 -E with the anchored key), take everything after
    # the first "=" via ${line#*=}, strip a trailing CR (CRLF files), trim surrounding
    # whitespace, then strip one matching pair of quotes.
    line="$(grep -m1 -E '^[[:space:]]*ORACLE_SCAN_ROOT[[:space:]]*=' "${env_file}" || true)"
    if [ -n "${line}" ]; then
      value="${line#*=}"
      value="${value%$'\r'}"
      # Trim leading/trailing whitespace.
      value="${value#"${value%%[![:space:]]*}"}"
      value="${value%"${value##*[![:space:]]}"}"
      case "${value}" in
        \"*\") value="${value#\"}"; value="${value%\"}" ;;
        \'*\') value="${value#\'}"; value="${value%\'}" ;;
      esac
      printf '%s\n' "${value}"
      return 0
    fi
  fi

  return 1
}

scan_root="$(resolve_scan_root)"
if [ -z "${scan_root:-}" ]; then
  echo "oracle-refresh: ORACLE_SCAN_ROOT is not set and no ORACLE_SCAN_ROOT= line was found in ${ORACLE_REFRESH_ENV_FILE:-${checkout_dir}/.env}" >&2
  exit 1
fi

if [ ! -d "${scan_root}" ]; then
  echo "oracle-refresh: scan root '${scan_root}' is not a directory" >&2
  exit 1
fi

pulled=0
uptodate=0
skipped=0
failed=0

if [ "${ORACLE_REFRESH_PULL:-1}" = "0" ]; then
  echo "oracle-refresh: ORACLE_REFRESH_PULL=0, skipping pull phase"
else
  for entry in "${scan_root}"/*/; do
    [ -d "${entry}" ] || continue
    repo_dir="${entry%/}"
    repo_name="$(basename -- "${repo_dir}")"

    if [ ! -e "${repo_dir}/.git" ]; then
      continue
    fi

    status_output=""
    if ! status_output="$(git -C "${repo_dir}" status --porcelain 2>/dev/null)"; then
      echo "${repo_name}: skipped (unreadable)"
      skipped=$((skipped + 1))
      continue
    fi

    if [ -n "${status_output}" ]; then
      echo "${repo_name}: skipped (dirty)"
      skipped=$((skipped + 1))
      continue
    fi

    if ! git -C "${repo_dir}" symbolic-ref -q HEAD >/dev/null 2>&1; then
      echo "${repo_name}: skipped (detached)"
      skipped=$((skipped + 1))
      continue
    fi

    if ! git -C "${repo_dir}" rev-parse --abbrev-ref --symbolic-full-name '@{u}' >/dev/null 2>&1; then
      echo "${repo_name}: skipped (no-upstream)"
      skipped=$((skipped + 1))
      continue
    fi

    old_sha="$(git -C "${repo_dir}" rev-parse HEAD 2>/dev/null)"
    pull_err=""
    if pull_err="$(git -C "${repo_dir}" pull --ff-only --quiet 2>&1 >/dev/null)"; then
      new_sha="$(git -C "${repo_dir}" rev-parse HEAD 2>/dev/null)"
      if [ "${old_sha}" = "${new_sha}" ]; then
        echo "${repo_name}: up-to-date"
        uptodate=$((uptodate + 1))
      else
        echo "${repo_name}: pulled ${old_sha:0:7}..${new_sha:0:7}"
        pulled=$((pulled + 1))
      fi
    else
      echo "${repo_name}: failed"
      reason="$(printf '%s\n' "${pull_err}" | grep -m1 -v '^[[:space:]]*$' || true)"
      if [ -n "${reason}" ]; then
        echo "  reason: ${reason}"
      fi
      failed=$((failed + 1))
    fi
  done
fi

echo "refresh: pulled ${pulled}, up-to-date ${uptodate}, skipped ${skipped}, failed ${failed}"

index_cmd="${ORACLE_REFRESH_INDEX_CMD:-npm run index}"
bash -c "${index_cmd}"
exit $?
