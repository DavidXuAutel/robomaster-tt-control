#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-a25689@10.239.121.14}"
REMOTE_PORT="${REMOTE_PORT:-30879}"
AERIAL_FT_CACHE="${AERIAL_FT_CACHE:-/tmp/aerial_ft_cache}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: B0_CHECKPOINT=... TRAIN_SUBSET=... CORRECTION_SET=... \
  TEXT_EMBEDS=... COLLECTION_MANIFEST=... sync_b0_ft_to_4090.sh [--dry-run]

Optional: DATASET_STATS (defaults beside checkpoint), CODE_ROOT, REMOTE_HOST,
REMOTE_PORT, AERIAL_FT_CACHE.
EOF
}

case "${1:-}" in
  --dry-run) DRY_RUN=1 ;;
  -h|--help) usage; exit 0 ;;
  "") ;;
  *) usage >&2; exit 2 ;;
esac

: "${B0_CHECKPOINT:?Set B0_CHECKPOINT to step_004000.pt}"
: "${TRAIN_SUBSET:?Set TRAIN_SUBSET to the converted training subset}"
: "${CORRECTION_SET:?Set CORRECTION_SET to the DAgger correction set}"
: "${TEXT_EMBEDS:?Set TEXT_EMBEDS to the OpenFly text embedding cache}"
: "${COLLECTION_MANIFEST:?Set COLLECTION_MANIFEST to collection_manifest.json}"

DATASET_STATS="${DATASET_STATS:-$(dirname "$B0_CHECKPOINT")/dataset_stats.json}"
CODE_ROOT="${CODE_ROOT:-$REPO_ROOT}"
ACCEL_CONFIG="$SCRIPT_DIR/accelerate_zero2_opt_offload_2proc.yaml"
DS_CONFIG_DIR="$CODE_ROOT/scripts/ds_configs"
HYDRA_CONFIG_DIR="$CODE_ROOT/configs"

if [[ "$(basename "$B0_CHECKPOINT")" != "step_004000.pt" ]]; then
  echo "Refusing checkpoint other than step_004000.pt: $B0_CHECKPOINT" >&2
  exit 2
fi
if [[ ! "$AERIAL_FT_CACHE" =~ ^/tmp/[A-Za-z0-9._/-]+$ ]]; then
  echo "AERIAL_FT_CACHE must be an absolute path below /tmp" >&2
  exit 2
fi
if [[ ! "$REMOTE_PORT" =~ ^[0-9]+$ ]]; then
  echo "REMOTE_PORT must be numeric" >&2
  exit 2
fi

for file in "$B0_CHECKPOINT" "$DATASET_STATS" "$COLLECTION_MANIFEST" "$ACCEL_CONFIG"; do
  [[ -f "$file" ]] || { echo "Missing required file: $file" >&2; exit 2; }
done
for directory in "$TRAIN_SUBSET" "$CORRECTION_SET" "$TEXT_EMBEDS" "$HYDRA_CONFIG_DIR" "$DS_CONFIG_DIR"; do
  [[ -d "$directory" ]] || { echo "Missing required directory: $directory" >&2; exit 2; }
done
git -C "$CODE_ROOT" rev-parse --is-inside-work-tree >/dev/null

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/aerial-ft-sync.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT
COMMIT="$(git -C "$CODE_ROOT" rev-parse HEAD)"
ARCHIVE="$TMP_DIR/FastWAM-${COMMIT}.tar.gz"
COMMIT_FILE="$TMP_DIR/COMMIT"
MANIFEST="$TMP_DIR/SHA256SUMS"
printf '%s\n' "$COMMIT" > "$COMMIT_FILE"
git -C "$CODE_ROOT" archive --format=tar.gz -o "$ARCHIVE" HEAD

python3 - "$MANIFEST" \
  "$B0_CHECKPOINT" "model/step_004000.pt" \
  "$DATASET_STATS" "model/dataset_stats.json" \
  "$TRAIN_SUBSET" "repo/data/openfly_lerobot/train_subset" \
  "$CORRECTION_SET" "repo/data/openfly_lerobot/b0_dagger_correction" \
  "$TEXT_EMBEDS" "repo/data/text_embeds_cache/openfly" \
  "$COLLECTION_MANIFEST" "manifests/collection_manifest.json" \
  "$ARCHIVE" "code/$(basename "$ARCHIVE")" \
  "$COMMIT_FILE" "code/COMMIT" \
  "$HYDRA_CONFIG_DIR" "repo/configs" \
  "$DS_CONFIG_DIR" "repo/scripts/ds_configs" \
  "$ACCEL_CONFIG" "repo/experiments/aerial/scripts/$(basename "$ACCEL_CONFIG")" <<'PY'
import hashlib
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
pairs = zip(sys.argv[2::2], sys.argv[3::2])
rows = []
for source_text, destination_text in pairs:
    source = pathlib.Path(source_text)
    files = sorted(path for path in ([source] if source.is_file() else source.rglob("*")) if path.is_file())
    for path in files:
        suffix = "" if source.is_file() else "/" + path.relative_to(source).as_posix()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        rows.append(f"{digest.hexdigest()}  {destination_text}{suffix}")
output.write_text("\n".join(rows) + "\n", encoding="utf-8")
PY

echo "Target: ${REMOTE_HOST}:${AERIAL_FT_CACHE} (SSH port ${REMOTE_PORT})"
echo "Commit: $COMMIT"
echo "Manifest entries: $(wc -l < "$MANIFEST" | tr -d ' ')"
if (( DRY_RUN )); then
  echo "DRY RUN: no SSH or rsync commands executed"
  echo "Would verify: (cd $AERIAL_FT_CACHE && sha256sum -c SHA256SUMS)"
  exit 0
fi

INCOMING="${AERIAL_FT_CACHE}.incoming"
SSH=(ssh -p "$REMOTE_PORT" "$REMOTE_HOST")
RSYNC_SHELL="ssh -p $REMOTE_PORT"
"${SSH[@]}" "set -e; rm -rf -- '$INCOMING'; mkdir -p -- \
  '$INCOMING/model' '$INCOMING/code' '$INCOMING/manifests' \
  '$INCOMING/repo/data/openfly_lerobot/train_subset' \
  '$INCOMING/repo/data/openfly_lerobot/b0_dagger_correction' \
  '$INCOMING/repo/data/text_embeds_cache/openfly' \
  '$INCOMING/repo/configs' '$INCOMING/repo/scripts/ds_configs' \
  '$INCOMING/repo/experiments/aerial/scripts'"

rsync -a -e "$RSYNC_SHELL" "$B0_CHECKPOINT" "$REMOTE_HOST:$INCOMING/model/step_004000.pt"
rsync -a -e "$RSYNC_SHELL" "$DATASET_STATS" "$REMOTE_HOST:$INCOMING/model/dataset_stats.json"
rsync -a -e "$RSYNC_SHELL" "$ARCHIVE" "$COMMIT_FILE" "$REMOTE_HOST:$INCOMING/code/"
"${SSH[@]}" "tar -xzf '$INCOMING/code/$(basename "$ARCHIVE")' -C '$INCOMING/repo'"
rsync -a --delete -e "$RSYNC_SHELL" "$TRAIN_SUBSET/" "$REMOTE_HOST:$INCOMING/repo/data/openfly_lerobot/train_subset/"
rsync -a --delete -e "$RSYNC_SHELL" "$CORRECTION_SET/" "$REMOTE_HOST:$INCOMING/repo/data/openfly_lerobot/b0_dagger_correction/"
rsync -a --delete -e "$RSYNC_SHELL" "$TEXT_EMBEDS/" "$REMOTE_HOST:$INCOMING/repo/data/text_embeds_cache/openfly/"
rsync -a -e "$RSYNC_SHELL" "$COLLECTION_MANIFEST" "$REMOTE_HOST:$INCOMING/manifests/collection_manifest.json"
rsync -a --delete -e "$RSYNC_SHELL" "$HYDRA_CONFIG_DIR/" "$REMOTE_HOST:$INCOMING/repo/configs/"
rsync -a --delete -e "$RSYNC_SHELL" "$DS_CONFIG_DIR/" "$REMOTE_HOST:$INCOMING/repo/scripts/ds_configs/"
rsync -a -e "$RSYNC_SHELL" "$ACCEL_CONFIG" "$REMOTE_HOST:$INCOMING/repo/experiments/aerial/scripts/"
rsync -a -e "$RSYNC_SHELL" "$MANIFEST" "$REMOTE_HOST:$INCOMING/SHA256SUMS"

"${SSH[@]}" "set -e; \
  cd '$INCOMING'; sha256sum -c SHA256SUMS; \
  rm -rf -- '${AERIAL_FT_CACHE}.previous'; \
  if [ -e '$AERIAL_FT_CACHE' ]; then mv -- '$AERIAL_FT_CACHE' '${AERIAL_FT_CACHE}.previous'; fi; \
  mv -- '$INCOMING' '$AERIAL_FT_CACHE'"
echo "Sync and remote SHA256 verification completed"
