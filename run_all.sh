#!/usr/bin/env bash
set -euo pipefail

readonly REPOSITORY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly APUF_DATA_ROOT="${APUF_DATA_ROOT:-${HOME}/Programming/APUF_CMODS7/data}"

cd "$REPOSITORY_DIR"

room_temperature() {
  uv run python scripts/analysis_room_temperature.py \
    --baseline-db "$APUF_DATA_ROOT/baseline/baseline_merged.db" \
    --modeling-db "$APUF_DATA_ROOT/modeling/modeling_merged.db" \
    --ergodicity-db "$APUF_DATA_ROOT/ergodicity/ergodicity_merged.db" \
    --baseline-challenges data/challenges/baseline.npy \
    --modeling-challenges data/challenges/modeling.npy \
    --ergodicity-challenges data/challenges/ergodicity.npy \
    --output data/processed/room_temperature \
    --progress-every 1

  uv run python scripts/summarize_room_temperature.py \
    --artifacts data/processed/room_temperature \
    --output tables/room_temperature

  uv run python scripts/plot_room_temperature.py \
    --artifacts data/processed/room_temperature \
    --output figures/room_temperature \
    --dpi 300
}

latent_geometry() {
  uv run python scripts/analysis_latent_geometry.py \
    --modeling-db "$APUF_DATA_ROOT/modeling/modeling_merged.db" \
    --modeling-challenges data/challenges/modeling.npy \
    --models data/processed/room_temperature/models.parquet \
    --output data/processed/latent_geometry \
    --neighbors 5 10 25 50 \
    --pair-samples 250000 \
    --umap-neighbors 30 \
    --umap-min-dist 0.1 \
    --target-weight 0.7 \
    --seed 123456

  uv run python scripts/plot_latent_geometry.py \
    --artifacts data/processed/latent_geometry \
    --output figures/latent_geometry \
    --dpi 300
}

margin_bank() {
  uv run python scripts/analysis_margin_bank.py \
    --modeling-db "$APUF_DATA_ROOT/modeling/modeling_merged.db" \
    --modeling-challenges data/challenges/modeling.npy \
    --models data/processed/room_temperature/models.parquet \
    --output data/processed/margin_bank \
    --challenges-per-bin 1 2 4 8 16 32 64 \
    --margin-min -6 \
    --margin-max 6 \
    --margin-bins 48

  uv run python scripts/plot_margin_bank.py \
    --artifacts data/processed/margin_bank \
    --output figures/margin_bank \
    --dpi 300
}

usage() {
  printf '%s\n' \
    "Usage: $0 STUDY [STUDY ...]" \
    "" \
    "Available studies:" \
    "  room_temperature" \
    "  latent_geometry" \
    "  margin_bank"
}

if (( $# == 0 )); then
  usage
  exit 2
fi

for study in "$@"; do
  case "$study" in
    room_temperature) room_temperature ;;
    latent_geometry) latent_geometry ;;
    margin_bank) margin_bank ;;
    *)
      printf 'Unknown study: %s\n\n' "$study" >&2
      usage >&2
      exit 2
      ;;
  esac
done
