#!/bin/bash
# ============================================================
# Piper Pilot 10-Demo Recording Script
# ============================================================
# Records 10 clean demonstrations for ACT training.
#
# Demo distribution:
#   episode_000_center
#   episode_001_center
#   episode_002_center
#   episode_003_center
#   episode_004_center
#   episode_005_center
#   episode_006_left
#   episode_007_left
#   episode_008_right
#   episode_009_right
#
# Usage:
#   bash scripts/record_piper_pilot_10demo.sh
#
# Controls during recording:
#   SPACE — start/stop recording an episode
#   R     — discard current episode and restart
#   E     — enable follower arm
#   D     — disable follower arm
#   Q/ESC — quit
#
# Important:
#   - Record in order (center→left→right) so episode indices
#     match the intent mapping in the collection checklist.
#   - Press R to discard any failed demo before moving to the next.
#   - This script never deletes or overwrites an old dataset. If data
#     already exists, it asks before resuming.
#   - The robot does not execute an automatic policy here; the operator
#     controls collection and accepts/discards demos manually.
#   - After recording all 10, run the sanity check:
#       python3 scripts/check_pilot_dataset.py \
#           --dataset data/lerobot_dataset_piper_bottle_pilot_10demo/
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

DATASET_ROOT="${PROJECT_ROOT}/data/lerobot_dataset_piper_bottle_pilot_10demo"
DATASET_REPO="piper/pilot_10demo"

echo "================================================"
echo "  Piper Pilot 10-Demo Recorder"
echo "================================================"
echo "  Dataset root: ${DATASET_ROOT}"
echo "  Repo ID     : ${DATASET_REPO}"
echo "  Task mode   : full_pick_place"
echo ""
echo "  Demo plan:"
echo "    episode_000_center"
echo "    episode_001_center"
echo "    episode_002_center"
echo "    episode_003_center"
echo "    episode_004_center"
echo "    episode_005_center"
echo "    episode_006_left"
echo "    episode_007_left"
echo "    episode_008_right"
echo "    episode_009_right"
echo ""
echo "  Controls:"
echo "    SPACE = start/stop recording"
echo "    R     = discard + restart current episode"
echo "    Q/ESC = quit"
echo "================================================"
echo ""

cd "${PROJECT_ROOT}"

# Use the piper_act conda environment
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate piper_act
fi

# Check that the dataset root doesn't already have episode data
# (we want a fresh start for the 10-demo pilot)
if [ -f "${DATASET_ROOT}/meta/info.json" ]; then
    EXISTING_EPISODES=$(python3 -c "import json; d=json.load(open('${DATASET_ROOT}/meta/info.json')); print(d.get('total_episodes',0))" 2>/dev/null || echo "0")
    if [ "${EXISTING_EPISODES}" != "0" ]; then
        echo "[WARN] Dataset already has ${EXISTING_EPISODES} episodes."
        echo "       The recorder will RESUME this dataset."
        echo "       If you want a fresh start, delete the dataset folder first."
        echo ""
        read -rp "       Continue with existing dataset? [y/N] " ANSWER
        if [ "${ANSWER}" != "y" ] && [ "${ANSWER}" != "Y" ]; then
            echo "Exiting."
            exit 0
        fi
    fi
fi

python3 teleop/data_collector.py \
    --task-mode full_pick_place \
    --dataset-root "${DATASET_ROOT}" \
    --dataset-repo-id "${DATASET_REPO}" \
    --record-gripper-action true

echo ""
echo "Recording session complete."
echo ""
echo "Next step: run sanity check"
echo "  python3 scripts/check_pilot_dataset.py --dataset ${DATASET_ROOT}"
