#!/usr/bin/env bash
set -euo pipefail

# Official LeRobot ACT training entry for the 40-episode single-cube dataset.
# This is intentionally separate from training/train.sh and the old dim256/chunk10 experiment.
#
# Dry-run / print command:
#   bash training/train_official_act_single_cube_40.sh
#
# Start training explicitly:
#   START_TRAINING=1 bash training/train_official_act_single_cube_40.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/huatec/miniconda3/envs/piper_act/bin/python}"
CONFIG="${CONFIG:-configs/train_official_act_single_cube_40.json}"
TASK_NAME="${TASK_NAME:-official_act_single_cube_40}"
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/data/single_cube_line4pos_40_clean}"
DATASET_REPO_ID="${DATASET_REPO_ID:-piper/single_cube_line4pos_40_clean}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/outputs/train/$TASK_NAME}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs/train}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${TASK_NAME}.log}"

EPISODES="${EPISODES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39}"
EXPECTED_EPISODES="${EXPECTED_EPISODES:-40}"
STEPS="${STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
LOG_FREQ="${LOG_FREQ:-100}"
SEED="${SEED:-1000}"
START_TRAINING="${START_TRAINING:-0}"

export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH=
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

EP_JSON="$("$PYTHON_BIN" - <<'PY' "$EPISODES"
import sys
episodes = [int(x) for x in sys.argv[1].split(",") if x.strip()]
print(str(episodes).replace(" ", ""))
PY
)"

echo "=== $TASK_NAME preflight at $(date) ==="
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "CONFIG=$CONFIG"
echo "DATASET_ROOT=$DATASET_ROOT"
echo "DATASET_REPO_ID=$DATASET_REPO_ID"
echo "EPISODES=$EP_JSON"
echo "OUT_DIR=$OUT_DIR"
echo "STEPS=$STEPS BATCH_SIZE=$BATCH_SIZE NUM_WORKERS=$NUM_WORKERS SAVE_FREQ=$SAVE_FREQ"
echo "policy=official_lerobot_act"
echo "architecture=chunk100_dim512_heads8_encoder4_decoder1_resnet18_imagenet"
echo "image_transforms=false"
echo "START_TRAINING=$START_TRAINING"
echo "no_hardware_access=true"

if [ ! -f "$CONFIG" ]; then
  echo "ERROR: missing config: $CONFIG" >&2
  exit 1
fi

if [ ! -d "$DATASET_ROOT" ]; then
  echo "WARN: dataset is not present yet: $DATASET_ROOT" >&2
  echo "      Use scripts/sync_official_act_from_5090.sh or set DATASET_ROOT." >&2
  if [ "$START_TRAINING" = "1" ]; then
    exit 1
  fi
else
  DATASET_ROOT="$DATASET_ROOT" EP_JSON="$EP_JSON" EXPECTED_EPISODES="$EXPECTED_EPISODES" "$PYTHON_BIN" - <<'PY'
from collections import Counter
from pathlib import Path
import json
import os

import pyarrow.parquet as pq

root = Path(os.environ["DATASET_ROOT"])
episodes = json.loads(os.environ["EP_JSON"])
expected = int(os.environ["EXPECTED_EPISODES"])
if len(episodes) != expected:
    raise SystemExit(f"ERROR: expected {expected} episodes, got {len(episodes)}")

info = json.loads((root / "meta" / "info.json").read_text())
required = [
    "observation.state",
    "action",
    "observation.images.global_rgb",
    "observation.images.wrist_rgb",
]
missing = [key for key in required if key not in info.get("features", {})]
if missing:
    raise SystemExit(f"ERROR: missing features: {missing}")

episode_rows = pq.read_table(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet").to_pylist()
selected = [row for row in episode_rows if int(row["episode_index"]) in episodes]
missing_eps = sorted(set(episodes) - {int(row["episode_index"]) for row in selected})
if missing_eps:
    raise SystemExit(f"ERROR: selected episodes missing: {missing_eps}")
lengths = [int(row["length"]) for row in selected]

data_table = pq.read_table(root / "data" / "chunk-000" / "file-000.parquet", columns=["episode_index"])
counts = Counter(int(row["episode_index"]) for row in data_table.to_pylist())

print("dataset_total_episodes=", info.get("total_episodes"))
print("dataset_total_frames=", info.get("total_frames"))
print("selected_episode_count=", len(selected))
print("selected_total_frames=", sum(lengths))
print("selected_frame_minmax=", min(lengths), max(lengths))
print("parquet_episode_count=", len(counts))
print("state_shape=", info["features"]["observation.state"]["shape"])
print("action_shape=", info["features"]["action"]["shape"])
PY
fi

if [ "$START_TRAINING" = "1" ] && [ -e "$OUT_DIR" ]; then
  echo "ERROR: output directory already exists: $OUT_DIR" >&2
  echo "Choose a new TASK_NAME/OUT_DIR or move the existing output before training." >&2
  exit 1
fi

TRAIN_CMD=(
  "$PYTHON_BIN" -u -m lerobot.scripts.lerobot_train
  --config_path "$CONFIG"
  --dataset.repo_id="$DATASET_REPO_ID"
  --dataset.root="$DATASET_ROOT"
  --dataset.episodes="$EP_JSON"
  --dataset.video_backend=pyav
  --dataset.image_transforms.enable=false
  --dataset.use_imagenet_stats=true
  --policy.type=act
  --policy.device=cuda
  --policy.push_to_hub=false
  --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1
  --output_dir="$OUT_DIR"
  --job_name="$TASK_NAME"
  --steps="$STEPS"
  --batch_size="$BATCH_SIZE"
  --num_workers="$NUM_WORKERS"
  --save_checkpoint=true
  --save_freq="$SAVE_FREQ"
  --log_freq="$LOG_FREQ"
  --seed="$SEED"
  --wandb.enable=false
)

printf 'training_command='
printf '%q ' "${TRAIN_CMD[@]}"
printf '\n'

if [ "$START_TRAINING" != "1" ]; then
  echo "training_not_started=true"
  echo "To start: START_TRAINING=1 bash training/$(basename "$0")"
  exit 0
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

echo "=== $TASK_NAME training started at $(date) ==="
"${TRAIN_CMD[@]}"
echo "=== $TASK_NAME finished at $(date) exit=$? ==="
