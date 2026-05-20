#!/bin/bash
# Diffusion Policy training for Piper bottle grasp.
#
# Prerequisites:
#   1. Dataset built with scripts/build_diffusion_dataset.py
#   2. Gripper units confirmed consistent (test_gripper_control.py)
#   3. ACT waypoints confirmed (quick_bottle_grasp.py succeeds)
#
# Smoke test (1K steps, small model):
#   STEPS=1000 SMOKE=1 bash training/train_diffusion_policy.sh
#
# Full training:
#   bash training/train_diffusion_policy.sh

set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-data/diffusion_dataset_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/piper_bottle_grasp_diffusion_v1}"
STEPS="${STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SMOKE="${SMOKE:-0}"

if [ "$SMOKE" = "1" ]; then
    STEPS=1000
    BATCH_SIZE=2
    HORIZON=8
    N_ACTION_STEPS=4
    DOWN_DIMS="128,256,512"
    OUTPUT_DIR="${OUTPUT_DIR}_smoke"
    echo "=== SMOKE TEST MODE ==="
    echo "  Steps: $STEPS"
    echo "  Batch: $BATCH_SIZE"
    echo "  Horizon: $HORIZON"
    echo "  n_action_steps: $N_ACTION_STEPS"
    echo "  Down dims: $DOWN_DIMS"
else
    HORIZON=16
    N_ACTION_STEPS=8
    DOWN_DIMS="512,1024,2048"
fi

echo "========================================"
echo "  Diffusion Policy Training"
echo "========================================"
echo "  Dataset: $DATASET_ROOT"
echo "  Output:  $OUTPUT_DIR"
echo "  Steps:   $STEPS"
echo "  Batch:   $BATCH_SIZE"
echo ""

PYTHONPATH= ~/miniconda3/envs/piper_act/bin/python3 -m lerobot.scripts.lerobot_train \
    --dataset.repo_id=piper/bottle_grasp_diffusion_v1 \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.image_transforms.enable=true \
    --policy.type=diffusion \
    --policy.horizon="${HORIZON}" \
    --policy.n_action_steps="${N_ACTION_STEPS}" \
    --policy.n_obs_steps=2 \
    --policy.down_dims="${DOWN_DIMS}" \
    --policy.num_train_timesteps=100 \
    --policy.num_inference_steps=10 \
    --policy.vision_backbone=resnet18 \
    --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
    --policy.use_separate_rgb_encoder_per_camera=true \
    --policy.optimizer_lr=1e-4 \
    --policy.optimizer_weight_decay=1e-6 \
    --policy.scheduler_name=cosine \
    --policy.scheduler_warmup_steps=500 \
    --policy.repo_id=piper/bottle_grasp_diffusion_v1 \
    --policy.push_to_hub=false \
    --batch_size="${BATCH_SIZE}" \
    --steps="${STEPS}" \
    --save_freq=5000 \
    --eval_freq=5000 \
    --output_dir="${OUTPUT_DIR}" \
    --job_name=piper_diffusion_v1

echo ""
echo "Training complete."
echo "Checkpoint: ${OUTPUT_DIR}/checkpoints/last/pretrained_model"
