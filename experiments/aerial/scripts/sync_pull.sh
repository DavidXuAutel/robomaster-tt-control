#!/usr/bin/env bash
# Aerial WAM — remote-host force-align pull (H100, or a fresh 4090 clone).
# Fetches all remotes and hard-aligns the local branch to the remote tip, so a
# host that fell behind (or split a pasted multi-line command and never ran the
# checkout) lands EXACTLY on what the Mac pushed — no merge, no fast-forward
# guesswork. Self-locating via `git rev-parse`.
#
# Usage:
#   experiments/aerial/scripts/sync_pull.sh                 # align to aerial-rl-skeleton
#   experiments/aerial/scripts/sync_pull.sh <branch>        # align to another branch
#
# Prefers remote `origin` (4090 bare repo, same LAN as H100 → fast), falls back
# to `github`. Refuses to clobber a dirty working tree unless FORCE=1 is set.
set -euo pipefail

BRANCH="${1:-aerial-rl-skeleton}"
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

echo "[sync_pull] repo:   $ROOT"
echo "[sync_pull] target: $BRANCH"

# Guard: a hard reset would silently discard local edits. Bail unless FORCE=1.
if ! git diff --quiet || ! git diff --cached --quiet; then
  if [[ "${FORCE:-0}" != "1" ]]; then
    echo "[sync_pull] ERROR: working tree is DIRTY — refusing to force-align." >&2
    echo "[sync_pull]   commit/stash first, or re-run with FORCE=1 to discard:" >&2
    echo "[sync_pull]   FORCE=1 $0 $BRANCH" >&2
    git status --short >&2
    exit 1
  fi
  echo "[sync_pull] WARNING: dirty tree + FORCE=1 → discarding local changes."
fi

echo "[sync_pull] fetching all remotes ..."
git fetch --all --prune

# Pick the remote to align to: origin (4090, LAN) first, then github.
REMOTE=""
for cand in origin github; do
  if git rev-parse --verify --quiet "refs/remotes/$cand/$BRANCH" >/dev/null; then
    REMOTE="$cand"
    break
  fi
done
if [[ -z "$REMOTE" ]]; then
  echo "[sync_pull] ERROR: neither origin/$BRANCH nor github/$BRANCH exists after fetch." >&2
  echo "[sync_pull]   remotes: $(git remote | tr '\n' ' ')" >&2
  exit 1
fi

TARGET_SHA="$(git rev-parse --short "$REMOTE/$BRANCH")"
echo "[sync_pull] aligning to $REMOTE/$BRANCH ($TARGET_SHA) ..."
git checkout -B "$BRANCH" "$REMOTE/$BRANCH"

echo "[sync_pull] HEAD now: $(git rev-parse --short HEAD)  ($BRANCH)  <- $REMOTE"
echo "[sync_pull] recent commits:"
git --no-pager log --oneline -3
