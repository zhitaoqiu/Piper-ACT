#!/usr/bin/env bash
# =============================================================================
# 机械臂复位脚本 —— 走到训练数据起始位姿后退出
# =============================================================================
# 用法:
#   ./scripts/reset_arm.sh                        用默认 checkpoint
#   CHECKPT=outputs/train/xxx/... ./scripts/reset_arm.sh   指定 checkpoint
# =============================================================================
set -euo pipefail

CHECKPT="${CHECKPT:-outputs/train/act_cube_approach64_global_current_d256_run1/checkpoints/last/pretrained_model}"

python3 inference/deploy.py \
    --checkpt "$CHECKPT" \
    --reset-to-recorded-start \
    --no-wrist \
    --test-mode A
