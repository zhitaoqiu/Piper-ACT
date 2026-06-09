#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# LeRobot ACT 训练
# ============================================================
# 用法:
#   ./training/train.sh                       默认配置训练
#   STEPS=8000 ./training/train.sh            改步数
#   BATCH_SIZE=8 SEED=42 ./training/train.sh  改批次+种子
#
# 断点续训:
#   RESUME=1 ./training/train.sh
# ============================================================

cd "$(dirname "$0")/.."

# ── 配置文件 ──────────────────────────────────────────────
CONFIG="${CONFIG:-configs/train_cube_64_dual_d256.json}"

# ── 断点续训 ──────────────────────────────────────────────
if [[ -n "${RESUME:-}" ]]; then
    RESUME_ARGS="--resume true --checkpoint_path outputs/train/act_cube_64_dual_d256/checkpoints/last/pretrained_model"
else
    RESUME_ARGS=""
fi

# ── 训练 ──────────────────────────────────────────────────
lerobot-train \
    --config_path "$CONFIG" \
    --steps="${STEPS:-15000}" \
    --batch_size="${BATCH_SIZE:-4}" \
    --save_freq="${SAVE_FREQ:-1000}" \
    --log_freq="${LOG_FREQ:-100}" \
    --seed="${SEED:-1000}" \
    --policy.optimizer_lr="${LR:-0.0003}" \
    $RESUME_ARGS
