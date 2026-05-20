#!/bin/bash
# Single-episode overfit sanity check for delta_horizon=1, no phase.
# Build data/lerobot_dataset_v2_delta_ep0 first with:
#   python3 scripts/rebuild_delta_phase_dataset.py \
#       --input-root data/lerobot_dataset_v2 --output-root data/lerobot_dataset_v2_delta_ep0 \
#       --delta-horizon-frames 1 --no-phase --episode 0 --overwrite

set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-data/lerobot_dataset_v2_delta_ep0}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/piper_bottle_grasp_delta_v2_overfit_ep0}"

PYTHONPATH= ~/miniconda3/envs/piper_act/bin/python3 -m lerobot.scripts.lerobot_train \
    --dataset.repo_id=piper/bottle_grasp_delta_phase \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.image_transforms.enable=false \
    --policy.type=act \
    --policy.chunk_size=10 \
    --policy.n_action_steps=1 \
    --policy.dim_model=512 \
    --policy.dim_feedforward=2048 \
    --policy.n_heads=8 \
    --policy.n_encoder_layers=4 \
    --policy.n_decoder_layers=4 \
    --policy.dropout=0.0 \
    --policy.use_vae=false \
    --policy.kl_weight=1.0 \
    --policy.optimizer_lr=1e-4 \
    --policy.optimizer_lr_backbone=1e-5 \
    --policy.repo_id=piper/bottle_grasp_act_delta_v2_overfit \
    --policy.push_to_hub=false \
    --batch_size=2 \
    --steps=5000 \
    --save_freq=1000 \
    --eval_freq=1000 \
    --output_dir="${OUTPUT_DIR}" \
    --job_name=piper_act_delta_v2_overfit_ep0
