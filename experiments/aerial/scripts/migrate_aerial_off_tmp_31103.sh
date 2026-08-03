#!/usr/bin/env bash
set -euo pipefail
LOG=/home/a25689/migrate_tmp_to_home_31103.log
STATUS=/home/a25689/migrate_tmp_to_home_31103.status
exec >>"$LOG" 2>&1
echo "=== BEGIN $(date -Is) ==="
printf 'RUNNING\n' >"$STATUS"

pkill -f 'experiments.aerial.orchestration.b1_discover' || true
pkill -f 'orch_b1_train.sh' || true
pkill -f 'scripts/train.py.*aerial_joint_b0_ft_dagger' || true
sleep 2

HOME_FT=/home/a25689/aerial_ft_cache
HOME_CACHE=/home/a25689/aerial_cache
HOME_EVAL=/home/a25689/aerial_eval_cache
mkdir -p "$HOME_FT" "$HOME_CACHE" "$HOME_EVAL"

if [[ -d /tmp/aerial_ft_cache && ! -L /tmp/aerial_ft_cache ]]; then
  echo 'rsync ft...'
  rsync -a /tmp/aerial_ft_cache/ "$HOME_FT/"
fi
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

ts=$(date +%Y%m%d-%H%M%S)
if [[ -d /tmp/aerial_ft_cache && ! -L /tmp/aerial_ft_cache ]]; then
  mv /tmp/aerial_ft_cache "/tmp/aerial_ft_cache.bak-$ts"
  ln -s "$HOME_FT" /tmp/aerial_ft_cache
fi
if [[ -d /tmp/aerial_cache && ! -L /tmp/aerial_cache ]]; then
  mv /tmp/aerial_cache "/tmp/aerial_cache.bak-$ts"
  ln -s "$HOME_CACHE" /tmp/aerial_cache
fi
if [[ -d /tmp/aerial_eval_cache && ! -L /tmp/aerial_eval_cache ]]; then
  mv /tmp/aerial_eval_cache "/tmp/aerial_eval_cache.bak-$ts"
  ln -s "$HOME_EVAL" /tmp/aerial_eval_cache
fi

# Drop local tmp backups once durable copy + symlink are verified.
if [[ -L /tmp/aerial_ft_cache && -f $HOME_FT/model/baseline.pt ]]; then
  rm -rf "/tmp/aerial_ft_cache.bak-$ts"
fi
if [[ -L /tmp/aerial_cache ]]; then
  rm -rf "/tmp/aerial_cache.bak-$ts"
fi
if [[ -L /tmp/aerial_eval_cache ]]; then
  rm -rf "/tmp/aerial_eval_cache.bak-$ts" 2>/dev/null || true
fi

echo '=== links ==='
ls -ld /tmp/aerial_ft_cache /tmp/aerial_cache /tmp/aerial_eval_cache || true
du -sh "$HOME_FT" "$HOME_CACHE" "$HOME_EVAL" || true
printf 'COMPLETED\n' >"$STATUS"
echo "=== END $(date -Is) ==="
