#!/bin/bash
# ============================================================
# ACT Pilot Training — 10-Demo Piper Bottle Grasp
# ============================================================
# Prerequisites:
#   1. 10 clean demos recorded in data/lerobot_dataset_piper_bottle_pilot_10demo/
#   2. Dataset passes sanity check (scripts/check_pilot_dataset.py)
#   3. conda activate piper_act
#
# Training output: outputs/train/act_pilot_10demo/
#
# Config: Tiny ACT (dim_model=128) with image transforms enabled.
# Uses 10 FPS LeRobot data with ~200 frames per episode.
# chunk_size=10 gives ~1 second action horizon at 10 FPS.
#
# Usage:
#   bash scripts/train_act_pilot_10demo.sh
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/data/lerobot_dataset_piper_bottle_pilot_10demo}"
DATASET_REPO_ID="${DATASET_REPO_ID:-piper/pilot_10demo}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/train/act_pilot_10demo}"
STEPS="${STEPS:-5000}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-/tmp/piper_act_hf_cache}"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/miniconda3/envs/piper_act/bin/python3}"
ALLOW_EXISTING_OUTPUT="${ALLOW_EXISTING_OUTPUT:-0}"
DEVICE="${DEVICE:-cuda}"
ALLOW_CPU_TRAINING="${ALLOW_CPU_TRAINING:-0}"

export HF_HOME="${HF_HOME:-${HF_CACHE_ROOT}/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_CACHE_ROOT}/datasets}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"

echo "================================================"
echo "  ACT Pilot Training — 10-Demo Bottle Grasp"
echo "================================================"
echo "  Dataset : ${DATASET_ROOT}"
echo "  Repo ID : ${DATASET_REPO_ID}"
echo "  Output  : ${OUTPUT_DIR}"
echo "  Steps   : ${STEPS}"
echo "  Device  : ${DEVICE}"
echo "================================================"

if [ ! -f "${DATASET_ROOT}/meta/info.json" ] || [ ! -d "${DATASET_ROOT}/data" ]; then
    echo "[ERROR] Dataset is missing: ${DATASET_ROOT}" >&2
    echo "  Run scripts/record_piper_pilot_10demo.sh first." >&2
    exit 1
fi

if [ -d "${OUTPUT_DIR}/checkpoints" ] && [ "${ALLOW_EXISTING_OUTPUT}" != "1" ]; then
    echo "[ERROR] Output directory already contains checkpoints: ${OUTPUT_DIR}/checkpoints" >&2
    echo "        This script will not overwrite existing pilot checkpoints." >&2
    echo "        Set ALLOW_EXISTING_OUTPUT=1 only if you intentionally want to resume/reuse this output." >&2
    exit 1
fi

if [ "${DEVICE}" = "cpu" ] && [ "${ALLOW_CPU_TRAINING}" != "1" ]; then
    echo "[ERROR] Refusing CPU training by default; this pilot is intended to train on CUDA." >&2
    echo "        Use DEVICE=cpu ALLOW_CPU_TRAINING=1 only when you intentionally want a slow CPU run." >&2
    exit 1
fi

if [[ "${DEVICE}" == cuda* ]]; then
    CUDA_AVAILABLE="$("${PYTHON_BIN}" -c "import torch; print('1' if torch.cuda.is_available() else '0')")"
    if [ "${CUDA_AVAILABLE}" != "1" ]; then
        echo "[ERROR] DEVICE=${DEVICE}, but PyTorch reports CUDA is unavailable." >&2
        echo "        Do not fall back to CPU while other training is running." >&2
        echo "        Re-run later on a CUDA-visible shell, or set DEVICE=cpu ALLOW_CPU_TRAINING=1 intentionally." >&2
        exit 1
    fi
fi

# Print dataset summary
echo ""
echo "Dataset summary:"
"${PYTHON_BIN}" -c "
import json
info = json.load(open('${DATASET_ROOT}/meta/info.json'))
print(f\"  Episodes: {info.get('total_episodes', 'N/A')}\")
print(f\"  Frames:   {info.get('total_frames', 'N/A')}\")
print(f\"  FPS:      {info.get('fps', 'N/A')}\")
" 2>/dev/null || true
echo ""

cd "${PROJECT_ROOT}"

echo "Running required dataset sanity check..."
"${PYTHON_BIN}" scripts/check_pilot_dataset.py --dataset "${DATASET_ROOT}"
echo "Sanity check passed. Starting ACT training."
echo ""

# ── Tiny ACT config ──
# dim_model=128, chunk_size=10, n_action_steps=10
# Image transforms enabled for the 5+ episode multi-demo setting.
# use_vae=false for simplicity (single-camera, small model).
PYTHONPATH= "${PYTHON_BIN}" -m lerobot.scripts.lerobot_train \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.image_transforms.enable=true \
    --policy.type=act \
    --policy.chunk_size=10 \
    --policy.n_action_steps=10 \
    --policy.dim_model=128 \
    --policy.dim_feedforward=512 \
    --policy.n_heads=4 \
    --policy.n_encoder_layers=2 \
    --policy.n_decoder_layers=2 \
    --policy.dropout=0.0 \
    --policy.use_vae=false \
    --policy.kl_weight=1.0 \
    --policy.optimizer_lr=3e-4 \
    --policy.optimizer_lr_backbone=1e-4 \
    --policy.device="${DEVICE}" \
    --policy.repo_id="${DATASET_REPO_ID}" \
    --policy.push_to_hub=false \
    --batch_size=8 \
    --num_workers=0 \
    --persistent_workers=false \
    --steps="${STEPS}" \
    --save_freq=1000 \
    --eval_freq=1000 \
    --output_dir="${OUTPUT_DIR}" \
    --job_name=act_pilot_10demo

echo ""
echo "Training complete."
echo ""
echo "Checkpoints saved to: ${OUTPUT_DIR}/checkpoints/"
echo ""
echo "Save training metadata:"
mkdir -p "${OUTPUT_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
cat > "${OUTPUT_DIR}/training_metadata_${TIMESTAMP}.json" << EOF
{
  "dataset_root": "${DATASET_ROOT}",
  "dataset_repo_id": "${DATASET_REPO_ID}",
  "policy_type": "act",
  "chunk_size": 10,
  "n_action_steps": 10,
  "dim_model": 128,
  "dim_feedforward": 512,
  "use_vae": false,
  "batch_size": 8,
  "steps": ${STEPS},
  "timestamp": "${TIMESTAMP}"
}
EOF
echo "  Metadata saved to: ${OUTPUT_DIR}/training_metadata_${TIMESTAMP}.json"
