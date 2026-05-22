#!/bin/bash
# ============================================================
# ACT Training — Adapter v2 10-Demo Piper Bottle
# ============================================================
# Dataset:
#   data/lerobot_dataset_piper_bottle_adapter_v2_10demo/
#
# Output:
#   outputs/train/act_adapter_v2_10demo/
#
# Camera:
#   observation.images.global_rgb  (single camera, /dev/video6)
#
# This script trains on 10 clean adapter-v2 demos collected via
# record_adapter_v2_mirror.py with --start-guard-mode zone.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/data/lerobot_dataset_piper_bottle_adapter_v2_10demo}"
DATASET_REPO_ID="${DATASET_REPO_ID:-piper/adapter_v2_10demo}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/train/act_adapter_v2_10demo}"
STEPS="${STEPS:-3000}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-/tmp/piper_act_hf_cache}"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/miniconda3/envs/piper_act/bin/python3}"
ALLOW_EXISTING_OUTPUT="${ALLOW_EXISTING_OUTPUT:-0}"
DEVICE="${DEVICE:-cuda}"
ALLOW_CPU_TRAINING="${ALLOW_CPU_TRAINING:-0}"

export HF_HOME="${HF_HOME:-${HF_CACHE_ROOT}/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_CACHE_ROOT}/datasets}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"

echo "================================================"
echo "  ACT Adapter v2 10-Demo Training"
echo "================================================"
echo "  Dataset : ${DATASET_ROOT}"
echo "  Repo ID : ${DATASET_REPO_ID}"
echo "  Output  : ${OUTPUT_DIR}"
echo "  Steps   : ${STEPS}"
echo "  Camera  : observation.images.global_rgb"
echo "  Device  : ${DEVICE}"
echo "================================================"

if [ ! -f "${DATASET_ROOT}/meta/info.json" ] || [ ! -d "${DATASET_ROOT}/data" ]; then
    echo "[ERROR] Dataset is missing: ${DATASET_ROOT}" >&2
    echo "        Record 10 clean demos with scripts/record_adapter_v2_mirror.py first." >&2
    echo "" >&2
    echo "        Command:" >&2
    echo "        python3 scripts/record_adapter_v2_mirror.py \\" >&2
    echo "          --can-port can0 \\" >&2
    echo "          --global-camera /dev/video6 \\" >&2
    echo "          --start-pose-file config/adapter_v2_start_pose.json \\" >&2
    echo "          --start-guard-mode zone \\" >&2
    echo "          --dataset-root ${DATASET_ROOT}" >&2
    exit 1
fi

if [ -d "${OUTPUT_DIR}/checkpoints" ] && [ "${ALLOW_EXISTING_OUTPUT}" != "1" ]; then
    echo "[ERROR] Output directory already contains checkpoints: ${OUTPUT_DIR}/checkpoints" >&2
    echo "        Set ALLOW_EXISTING_OUTPUT=1 to resume/reuse." >&2
    exit 1
fi

if [ "${DEVICE}" = "cpu" ] && [ "${ALLOW_CPU_TRAINING}" != "1" ]; then
    echo "[ERROR] Refusing CPU training by default." >&2
    echo "        Use DEVICE=cpu ALLOW_CPU_TRAINING=1 intentionally." >&2
    exit 1
fi

if [[ "${DEVICE}" == cuda* ]]; then
    CUDA_AVAILABLE="$("${PYTHON_BIN}" -c "import torch; print('1' if torch.cuda.is_available() else '0')")"
    if [ "${CUDA_AVAILABLE}" != "1" ]; then
        echo "[ERROR] DEVICE=${DEVICE}, but PyTorch reports CUDA is unavailable." >&2
        exit 1
    fi
fi

cd "${PROJECT_ROOT}"

echo ""
echo "Running required dataset sanity check..."
"${PYTHON_BIN}" scripts/check_pilot_dataset.py \
    --dataset "${DATASET_ROOT}" \
    --expected-episodes 10 \
    --min-pass-episodes 10 \
    --require-single-camera \
    --camera-key observation.images.global_rgb
echo "Sanity check passed. Starting ACT training."
echo ""

PYTHONPATH= "${PYTHON_BIN}" -m lerobot.scripts.lerobot_train \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.image_transforms.enable=false \
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
    --job_name=act_adapter_v2_10demo

echo ""
echo "Training complete."
echo "Checkpoints saved to: ${OUTPUT_DIR}/checkpoints/"
