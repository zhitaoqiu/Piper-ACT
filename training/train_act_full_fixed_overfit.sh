#!/bin/bash
# ============================================================
# Full-Trajectory ACT Training — Fixed Position Overfit
# ============================================================
# Data collected via:
#   python3 teleop/data_collector.py \
#     --task-mode full_pick_place \
#     --dataset-root data/lerobot_dataset_full_fixed_1ep \
#     --dataset-repo-id piper/bottle_full_fixed_1ep
#
# Trained policy outputs 7D absolute actions [J1..J6, gripper].
# gripper is NOT forced open — model learns close/release timing.
# ============================================================
# Usage:
#   bash training/train_act_full_fixed_overfit.sh
# ============================================================
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-data/lerobot_dataset_full_fixed_1ep}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/act_full_fixed_overfit}"
STEPS="${STEPS:-5000}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-/tmp/piper_act_hf_cache}"

export HF_HOME="${HF_HOME:-${HF_CACHE_ROOT}/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_CACHE_ROOT}/datasets}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"

if [[ ! -f "${DATASET_ROOT}/meta/info.json" || ! -d "${DATASET_ROOT}/data" ]]; then
    echo "[ERROR] Dataset is missing: ${DATASET_ROOT}" >&2
    echo "  Run data_collector.py with --task-mode full_pick_place first." >&2
    exit 1
fi

# Tiny ACT for overfit — small model, fast train, 7D action incl. gripper
# chunk_size=50 provides temporal structure to avoid mean-action collapse
PYTHONPATH= ~/miniconda3/envs/piper_act/bin/python3 -m lerobot.scripts.lerobot_train \
    --dataset.repo_id="${DATASET_REPO_ID:-piper/bottle_full_fixed_1ep}" \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.image_transforms.enable=false \
    --policy.type=act \
    --policy.chunk_size=50 \
    --policy.n_action_steps=50 \
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
    --policy.repo_id=piper/bottle_full_fixed_overfit \
    --policy.push_to_hub=false \
    --batch_size=8 \
    --num_workers=0 \
    --persistent_workers=false \
    --steps="${STEPS}" \
    --save_freq=1000 \
    --eval_freq=1000 \
    --output_dir="${OUTPUT_DIR}" \
    --job_name=act_full_fixed_overfit

# ============================================================
# Future: standard ACT config (from expert/lerobot_piper3)
# For multi-episode, multi-position generalization.
# DO NOT enable until 1ep overfit passes offline checks.
#
#   --policy.chunk_size=100 \
#   --policy.n_action_steps=100 \
#   --policy.dim_model=512 \
#   --policy.dim_feedforward=3200 \
#   --policy.n_heads=8 \
#   --policy.n_encoder_layers=4 \
#   --policy.n_decoder_layers=1 \
#   --policy.use_vae=true \
#   --policy.latent_dim=32 \
#   --policy.kl_weight=10.0 \
#   --policy.dropout=0.1 \
#   --policy.optimizer_lr=1e-5 \
#   --batch_size=4 \
#   --steps=50000 \
# ============================================================
