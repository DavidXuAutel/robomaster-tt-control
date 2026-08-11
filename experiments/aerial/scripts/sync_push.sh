#!/usr/bin/env bash
# Aerial WAM — Mac-side push: commit (optional) + push the current branch to
# ALL remotes (github mirror + origin = 4090 bare repo). Self-locating: cd's to
# the repo root via `git rev-parse`, so the SAME script runs from any checkout
# on any host with no hardcoded paths.
#
# Usage:
#   experiments/aerial/scripts/sync_push.sh                 # push only (no commit)
#   experiments/aerial/scripts/sync_push.sh "commit msg"    # add -A, commit, push
#
# Why this exists: the multi-remote push cycle is error-prone by hand — pushing
# from the wrong repo dir (~/Projects vs the worktree) gives "src refspec ...
# does not match any", and forgetting one remote leaves H100 pulling a stale SHA.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
BRANCH="$(git branch --show-current)"
if [[ -z "$BRANCH" ]]; then
  echo "[sync_push] ERROR: detached HEAD — checkout a branch first." >&2
  exit 1
fi

echo "[sync_push] repo:   $ROOT"
echo "[sync_push] branch: $BRANCH"

# Optional commit: only when a message is given AND there is something staged/dirty.
if [[ $# -gt 0 ]]; then
  MSG="$*"
  git add -A
  if git diff --cached --quiet; then
    echo "[sync_push] nothing to commit (working tree already clean)."
  else
    git commit -m "$MSG"
    echo "[sync_push] committed: $MSG"
  fi
fi

REMOTES="$(git remote)"
if [[ -z "$REMOTES" ]]; then
  echo "[sync_push] ERROR: no git remotes configured." >&2
  exit 1
fi

RC=0
for r in $REMOTES; do
  echo "[sync_push] --> pushing $BRANCH to $r ..."
  if git push "$r" "$BRANCH"; then
    echo "[sync_push]     ok: $r"
  else
    echo "[sync_push]     FAILED: $r (network? auth? see above)" >&2
    RC=1
  fi
done

echo "[sync_push] local HEAD: $(git rev-parse --short HEAD)  ($BRANCH)"
if [[ $RC -ne 0 ]]; then
  echo "[sync_push] one or more pushes FAILED — remotes are NOT all in sync." >&2
fi
exit $RC
