#!/usr/bin/env bash
# =============================================================================
# 机械臂复位脚本 —— 走到训练数据起始位姿后退出
# =============================================================================
# 用法:
#   ./scripts/reset_arm.sh                        用默认 checkpoint
#   CHECKPT=outputs/train/xxx/... ./scripts/reset_arm.sh   指定 checkpoint
# =============================================================================
set -euo pipefail

CHECKPT="${CHECKPT:-outputs/act_single_cube_40_official_resnet18/checkpoints/100000/pretrained_model}"

# SmolVLA VERIFIED_START_QPOS
START_QPOS="--start-qpos 0.02430,0.00670,-0.00390,0.01610,0.31150,-0.07480,0.09870"

python3 inference/deploy.py \
    --checkpt "$CHECKPT" \
    --reset-to-recorded-start \
    $START_QPOS \
    --no-gui \
    --test-mode A
