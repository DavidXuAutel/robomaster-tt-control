#!/usr/bin/env bash
set -euo pipefail
LOG=/home/a25689/migrate_tmp_to_home_30682.log
STATUS=/home/a25689/migrate_tmp_to_home_30682.status
exec >>"$LOG" 2>&1
echo "=== BEGIN $(date -Is) ==="
printf 'RUNNING\n' >"$STATUS"

HOME_CACHE=/home/a25689/aerial_cache
HOME_EVAL=/home/a25689/aerial_eval_cache
mkdir -p "$HOME_CACHE" "$HOME_EVAL"

# Do not kill orch_eval_worker mid-job if possible; only migrate trees.
if [[ -d /tmp/aerial_cache && ! -L /tmp/aerial_cache ]]; then
  echo 'rsync cache...'
  rsync -a /tmp/aerial_cache/ "$HOME_CACHE/"
fi
if [[ -d /tmp/aerial_eval_cache && ! -L /tmp/aerial_eval_cache ]]; then
  echo 'rsync eval...'
  rsync -a /tmp/aerial_eval_cache/ "$HOME_EVAL/"
fi

mkdir -p /home/a25689/aerial_cache_shared/orchestration
if [[ -d $HOME_CACHE/orchestration ]]; then
  rsync -a "$HOME_CACHE/orchestration/" /home/a25689/aerial_cache_shared/orchestration/
fi
if [[ -d $HOME_EVAL/orchestration ]]; then
  rsync -a "$HOME_EVAL/orchestration/" /home/a25689/aerial_cache_shared/orchestration/ || true
fi

ts=$(date +%Y%m%d-%H%M%S)
if [[ -d /tmp/aerial_cache && ! -L /tmp/aerial_cache ]]; then
  mv /tmp/aerial_cache "/tmp/aerial_cache.bak-$ts"
  ln -s "$HOME_CACHE" /tmp/aerial_cache
fi
if [[ -d /tmp/aerial_eval_cache && ! -L /tmp/aerial_eval_cache ]]; then
  mv /tmp/aerial_eval_cache "/tmp/aerial_eval_cache.bak-$ts"
  ln -s "$HOME_EVAL" /tmp/aerial_eval_cache
fi

if [[ -L /tmp/aerial_cache ]]; then
  rm -rf "/tmp/aerial_cache.bak-$ts"
fi
if [[ -L /tmp/aerial_eval_cache ]]; then
  # eval worker may still have open files under the bak tree; keep bak if busy
  if ! lsof +D "/tmp/aerial_eval_cache.bak-$ts" >/dev/null 2>&1; then
    rm -rf "/tmp/aerial_eval_cache.bak-$ts"
  else
    echo "kept bak due to open files: /tmp/aerial_eval_cache.bak-$ts"
  fi
fi

echo '=== links ==='
ls -ld /tmp/aerial_cache /tmp/aerial_eval_cache || true
du -sh "$HOME_CACHE" "$HOME_EVAL" || true
printf 'COMPLETED\n' >"$STATUS"
echo "=== END $(date -Is) ==="
