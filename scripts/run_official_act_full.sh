#!/usr/bin/env bash
# =============================================================================
# Official ACT 全流程测试 —— 100K步模型，双摄，真实机械臂
# =============================================================================
# 模型：ACT ResNet18, single_cube_line4pos_40_clean, 100K步
# chunk_size=100, n_action_steps=100, 双摄 (global + wrist)
# =============================================================================
# 用法:
#   bash scripts/run_official_act_full.sh             默认参数
#   STEPS=500 bash scripts/run_official_act_full.sh   改步数
# =============================================================================

set -euo pipefail

CHECKPT="${CHECKPT:-outputs/act_single_cube_40_official_resnet18/checkpoints/100000/pretrained_model}"

POLICY_TYPE="act-full"
TEST_MODE="full-e2e"
STOP_AFTER="full"
BACKEND="direct_sdk"

STEPS="${STEPS:-500}"
CHUNK_EXEC="--act-full-chunk-exec target_reached"
GRIPPER="--open-gripper-on-start"

# SmolVLA VERIFIED_START_QPOS
START_QPOS="--start-qpos 0.02430,0.00670,-0.00390,0.01610,0.31150,-0.07480,0.09870"

# 最小干预：模型直通，但保留轻微平滑防抖
ACTION_SMOOTH="--action-smooth 0.4"
MAX_DELTA_ARM="--max-delta-arm 0.15"
MAX_DELTA_WRIST="--max-delta-wrist 0.08"
WRIST_FREEZE="--wrist-freeze-j2 2.5"
VELOCITY="--velocity-pct 35"

DEBUG_FLAGS="--debug-actions --debug-policy-io"
SAVE_FLAGS="--save-global-video --save-final-images"
ALLOW="--allow-real-full-e2e"

python3 inference/deploy.py \
    --policy-type "$POLICY_TYPE" \
    --checkpt "$CHECKPT" \
    --control-backend "$BACKEND" \
    --no-gui \
    --test-mode "$TEST_MODE" \
    --full-e2e-stop-after "$STOP_AFTER" \
    --approach-steps "$STEPS" \
    $CHUNK_EXEC \
    $GRIPPER \
    $START_QPOS \
    $ACTION_SMOOTH \
    $MAX_DELTA_ARM \
    $MAX_DELTA_WRIST \
    $WRIST_FREEZE \
    $VELOCITY \
    $DEBUG_FLAGS \
    $SAVE_FLAGS \
    $ALLOW \
    "$@"