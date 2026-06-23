#!/usr/bin/env bash
set -euo pipefail

# Copy the 5090 official ACT artifacts into this project.
# This script does not train and does not touch hardware.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="${REMOTE:-xvla-gpu}"
REMOTE_Q_WS="${REMOTE_Q_WS:-/root/autodl-tmp/q_ws}"

REMOTE_DATASET="${REMOTE_DATASET:-$REMOTE_Q_WS/datasets/single_cube_line4pos_40_clean/}"
REMOTE_CHECKPOINT="${REMOTE_CHECKPOINT:-$REMOTE_Q_WS/outputs/act_single_cube_40_official_resnet18/checkpoints/100000/pretrained_model/}"

LOCAL_DATASET="${LOCAL_DATASET:-$PROJECT_ROOT/data/single_cube_line4pos_40_clean/}"
LOCAL_CHECKPOINT="${LOCAL_CHECKPOINT:-$PROJECT_ROOT/outputs/train/official_act_single_cube_40/checkpoints/100000/pretrained_model/}"

echo "REMOTE=$REMOTE"
echo "REMOTE_DATASET=$REMOTE_DATASET"
echo "REMOTE_CHECKPOINT=$REMOTE_CHECKPOINT"
echo "LOCAL_DATASET=$LOCAL_DATASET"
echo "LOCAL_CHECKPOINT=$LOCAL_CHECKPOINT"
echo "no_hardware_access=true"

mkdir -p "$LOCAL_DATASET" "$LOCAL_CHECKPOINT"

rsync -a --info=progress2 "$REMOTE:$REMOTE_DATASET" "$LOCAL_DATASET"
rsync -a --info=progress2 "$REMOTE:$REMOTE_CHECKPOINT" "$LOCAL_CHECKPOINT"

echo "sync_done=true"
