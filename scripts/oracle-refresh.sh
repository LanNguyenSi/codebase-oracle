#!/usr/bin/env bash
# codebase-oracle: scheduled refresh for a machine that serves as the index
# source of truth.
#
# Fast-forwards every clean, non-detached checkout with an upstream that sits
# directly under ORACLE_SCAN_ROOT, then runs the incremental index. Meant to
# be driven by a scheduler (systemd timer, launchd job, cron) so the index
# host stays current without a human running `npm run index` by hand.
#
# A dirty checkout, a detached HEAD, or a branch with no upstream is skipped
# rather than touched; a failed pull is reported and the loop continues so
# one bad repo doesn't block the rest. The index step always runs afterward
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
#                           `npm run index`.
set -u

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
checkout_dir="$(cd -- "${script_dir}/.." >/dev/null 2>&1 && pwd -P)"
cd -- "${checkout_dir}" || exit 1

resolve_scan_root() {
  if [ -n "${ORACLE_SCAN_ROOT:-}" ]; then
    printf '%s\n' "${ORACLE_SCAN_ROOT}"
    return 0
  fi

  local env_file="${checkout_dir}/.env"
  if [ -f "${env_file}" ]; then
    local line value
    line="$(grep -m1 '^ORACLE_SCAN_ROOT=' "${env_file}" || true)"
    if [ -n "${line}" ]; then
      value="${line#ORACLE_SCAN_ROOT=}"
      # Strip a single layer of matching surrounding quotes.
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
  echo "oracle-refresh: ORACLE_SCAN_ROOT is not set and no ORACLE_SCAN_ROOT= line was found in ${checkout_dir}/.env" >&2
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

    if [ -n "$(git -C "${repo_dir}" status --porcelain 2>/dev/null)" ]; then
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
    if git -C "${repo_dir}" pull --ff-only --quiet 2>/dev/null; then
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
      failed=$((failed + 1))
    fi
  done
fi

echo "refresh: pulled ${pulled}, up-to-date ${uptodate}, skipped ${skipped}, failed ${failed}"

index_cmd="${ORACLE_REFRESH_INDEX_CMD:-npm run index}"
bash -c "${index_cmd}"
exit $?
