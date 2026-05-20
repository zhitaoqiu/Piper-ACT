#!/usr/bin/env python3
"""Numerically check whether an ACT policy collapsed to constant actions."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def set_hf_cache_defaults(cache_dir: Path) -> None:
    os.environ.setdefault("HF_HOME", str(cache_dir / "hf_home"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir / "datasets"))


def load_policy_processors(policy, checkpt: str, device: torch.device):
    from lerobot.policies.factory import make_pre_post_processors

    preprocessor_overrides = {
        "device_processor": {"device": device.type},
        "normalizer_processor": {"device": device.type},
    }
    postprocessor_overrides = {
        "unnormalizer_processor": {"device": device.type},
        "device_processor": {"device": "cpu"},
    }
    return make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpt,
        preprocessor_overrides=preprocessor_overrides,
        postprocessor_overrides=postprocessor_overrides,
    )


def build_batch(item: dict, device: torch.device) -> dict:
    batch = {}
    for key, value in item.items():
        if key == "observation.state" or key.startswith("observation.images."):
            batch[key] = value.unsqueeze(0).to(device)
    return batch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpt", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="piper/bottle_grasp")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/piper_act_hf_cache"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_hf_cache_defaults(args.cache_dir)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.modeling_act import ACTPolicy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root)
    episode_index = np.asarray(dataset.hf_dataset["episode_index"])
    positions = np.flatnonzero(episode_index == args.episode)
    if len(positions) == 0:
        raise SystemExit(f"Episode {args.episode} not found")
    positions = positions[: args.frames]

    policy = ACTPolicy.from_pretrained(args.checkpt).to(device).eval()
    preprocessor, postprocessor = load_policy_processors(policy, args.checkpt, device)

    preds = []
    gts = []
    chunk_ranges = []
    for pos in positions:
        item = dataset[int(pos)]
        with torch.inference_mode():
            norm = preprocessor(build_batch(item, device))
            chunk = policy.predict_action_chunk(norm)
            chunk = postprocessor(chunk).detach().cpu().numpy()[0]
        preds.append(chunk[0])
        gts.append(item["action"].numpy())
        chunk_ranges.append(float(np.max(np.ptp(chunk[:, :6], axis=0))))

    preds = np.asarray(preds)
    gts = np.asarray(gts)
    pred_step = np.max(np.abs(np.diff(preds[:, :6], axis=0)), axis=1) if len(preds) > 1 else np.array([0.0])
    gt_step = np.max(np.abs(np.diff(gts[:, :6], axis=0)), axis=1) if len(gts) > 1 else np.array([0.0])
    pred_range = np.ptp(preds[:, :6], axis=0)
    gt_range = np.ptp(gts[:, :6], axis=0)
    chunk_ranges = np.asarray(chunk_ranges)
    mse = float(np.mean((preds - gts) ** 2))

    print(f"checkpoint: {args.checkpt}")
    print(f"dataset: {args.dataset_root}, episode={args.episode}, frames={len(positions)}")
    print(f"mse: {mse:.6f}")
    print(
        "pred_step mean/p95/max:",
        f"{pred_step.mean():.6f}",
        f"{np.quantile(pred_step, 0.95):.6f}",
        f"{pred_step.max():.6f}",
    )
    print(
        "gt_step   mean/p95/max:",
        f"{gt_step.mean():.6f}",
        f"{np.quantile(gt_step, 0.95):.6f}",
        f"{gt_step.max():.6f}",
    )
    print("pred_range arm:", np.round(pred_range, 6).tolist())
    print("gt_range   arm:", np.round(gt_range, 6).tolist())
    print(
        "chunk_internal_arm_range mean/max:",
        f"{chunk_ranges.mean():.6f}",
        f"{chunk_ranges.max():.6f}",
    )
    collapsed = bool(pred_step.max() < 1e-4 and chunk_ranges.max() < 1e-4)
    print(f"collapsed: {collapsed}")
    return 2 if collapsed else 0


if __name__ == "__main__":
    raise SystemExit(main())
