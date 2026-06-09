#!/usr/bin/env bash
# =============================================================================
# Cube 方块抓取 —— 64条均衡 单摄 全流程
# =============================================================================
# 模型：ACT dim256，64条均衡（蓝/紫 × R0/R90），15000步
# 数据集：cube_64_global（全局相机，全流程：进近→闭合→返回）
# =============================================================================

set -euo pipefail

CHECKPT="outputs/train/act_cube_64_dual_d256/checkpoints/last/pretrained_model"

# act-full：模型输出控制夹爪（数据含闭合，和训练一致）
# 模型：ACT dim256，cube 64条 均衡（蓝/紫 × R0/R90），15000步
POLICY_TYPE="act-full"
TEST_MODE="full-e2e"
STOP_AFTER="full"

CAMERA=""
RESET_MODE=""

STEPS=300
REPLAN=""
CHUNK_EXEC="--act-full-chunk-exec target_reached"
GRIPPER="--open-gripper-on-start"
J2_CLAMP=""

DEBUG_FLAGS="--debug-actions --debug-policy-io"
SAVE_FLAGS="--save-global-video --save-final-images"
ALLOW="--allow-real-full-e2e"

python3 inference/deploy.py \
    --policy-type "$POLICY_TYPE" \
    --checkpt "$CHECKPT" \
    --test-mode "$TEST_MODE" \
    --full-e2e-stop-after "$STOP_AFTER" \
    $CAMERA \
    $RESET_MODE \
    --approach-steps "$STEPS" \
    $REPLAN \
    $CHUNK_EXEC \
    $GRIPPER \
    $J2_CLAMP \
    $DEBUG_FLAGS \
    $SAVE_FLAGS \
    $ALLOW
