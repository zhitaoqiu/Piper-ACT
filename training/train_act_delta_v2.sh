#!/bin/bash
# ACT training on delta_horizon=1 dataset, no phase.
# action = state[t+1] - state[t] (single-frame delta, 7D)

set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-data/lerobot_dataset_v2_delta}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/piper_bottle_grasp_delta_v2}"

PYTHONPATH= ~/miniconda3/envs/piper_act/bin/python3 -m lerobot.scripts.lerobot_train \
    --dataset.repo_id=piper/bottle_grasp_delta_phase \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.image_transforms.enable=true \
    --policy.type=act \
    --policy.chunk_size=10 \
    --policy.n_action_steps=1 \
    --policy.dim_model=512 \
    --policy.dim_feedforward=2048 \
    --policy.n_heads=8 \
    --policy.n_encoder_layers=4 \
    --policy.n_decoder_layers=4 \
    --policy.dropout=0.1 \
    --policy.use_vae=false \
    --policy.kl_weight=1.0 \
    --policy.optimizer_lr=1e-4 \
    --policy.optimizer_lr_backbone=1e-5 \
    --policy.repo_id=piper/bottle_grasp_act_delta_v2 \
    --policy.push_to_hub=false \
    --batch_size=4 \
    --steps=30000 \
    --save_freq=5000 \
    --eval_freq=5000 \
    --output_dir="${OUTPUT_DIR}" \
    --job_name=piper_act_delta_v2
