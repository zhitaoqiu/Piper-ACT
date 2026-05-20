#!/bin/bash
# Build a clean single-episode dataset for the current environment.
# This keeps old-environment episodes out of both samples and normalization stats.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/piper_act/bin/python3}"
SOURCE_ROOT="${SOURCE_ROOT:-data/lerobot_dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/lerobot_dataset_today_approach_1ep}"
EPISODE="${EPISODE:-latest}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-/tmp/piper_act_hf_cache}"
REPORT_PATH="${REPORT_PATH:-reports/today_approach_1ep_rebuild_report.csv}"
VCODEC="${VCODEC:-h264}"
ENCODER_THREADS="${ENCODER_THREADS:-2}"

export HF_HOME="${HF_HOME:-${HF_CACHE_ROOT}/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_CACHE_ROOT}/datasets}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"

if [[ "${EPISODE}" == "latest" ]]; then
    EPISODE="$(SOURCE_ROOT_FOR_PY="${SOURCE_ROOT}" "${PYTHON_BIN}" -c 'import os
from pathlib import Path
import pandas as pd
root = Path(os.environ["SOURCE_ROOT_FOR_PY"]) / "data"
paths = sorted(root.glob("chunk-*/file-*.parquet"))
if not paths:
    raise SystemExit(f"No parquet files found under {root}")
episodes = []
for path in paths:
    episodes.extend(pd.read_parquet(path, columns=["episode_index"])["episode_index"].unique().tolist())
print(max(int(ep) for ep in episodes))
')"
fi

echo "[BUILD] source dataset: ${SOURCE_ROOT}"
echo "[BUILD] source episode: ${EPISODE}"
echo "[BUILD] output dataset: ${OUTPUT_ROOT}"

"${PYTHON_BIN}" scripts/rebuild_trimmed_dataset.py \
    --input-root "${SOURCE_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --repo-id piper/bottle_grasp \
    --output-repo-id piper/bottle_approach_today_1ep \
    --episode "${EPISODE}" \
    --report-path "${REPORT_PATH}" \
    --cache-dir "${HF_CACHE_ROOT}" \
    --vcodec "${VCODEC}" \
    --encoder-threads "${ENCODER_THREADS}" \
    --overwrite
