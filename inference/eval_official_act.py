#!/usr/bin/env python3
"""Offline evaluation for the separate official LeRobot ACT route.

This script never connects to Piper hardware. It loads ACT checkpoints and a
LeRobot dataset, then compares predicted actions against recorded actions.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

JOINT_NAMES = ("j1", "j2", "j3", "j4", "j5", "j6", "gripper")
IMAGE_PREFIX = "observation.images."
STATE_KEY = "observation.state"
ACTION_KEY = "action"
PIPER_LIMITS = (
    (-2.6179, 2.6179),
    (0.0, 3.14),
    (-2.967, 0.0),
    (-1.745, 1.745),
    (-1.22, 1.22),
    (-2.09439, 2.09439),
    (0.0, None),
)


@dataclass
class EpisodeStats:
    checkpoint: str
    episode_index: int
    mode: str
    valid_frames: int
    total_frames: int
    errors: int
    nan_count: int
    limit_warnings: int
    mae: float
    per_frame_mae_max: float
    max_delta_norm: float
    gripper_pred_trend: float
    per_joint_mae: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline official ACT evaluation")
    parser.add_argument("--dataset-root", default="data/single_cube_line4pos_40_clean")
    parser.add_argument("--dataset-repo-id", default="piper/single_cube_line4pos_40_clean")
    parser.add_argument("--dataset-revision", default="main",
                        help="Dataset revision passed to LeRobotDataset. Use 'main' for local-only datasets to avoid Hub version lookup.")
    parser.add_argument("--checkpoint", nargs="+", required=True,
                        help="One or more ACT pretrained_model checkpoint directories.")
    parser.add_argument("--episode-indices", default=None,
                        help="Comma-separated episode indices. Default: last --episodes episodes.")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of final episodes to evaluate when --episode-indices is unset.")
    parser.add_argument("--max-frames", type=int, default=80,
                        help="Max frames per episode, 0 means all frames.")
    parser.add_argument("--mode", choices=("checkpoint_selection", "deployment_simulation"),
                        default="checkpoint_selection",
                        help="checkpoint_selection resets and takes first action of a fresh chunk each frame; "
                             "deployment_simulation keeps the ACT action queue.")
    parser.add_argument("--exec-horizon", type=int, default=None,
                        help="Runtime override for policy.config.n_action_steps.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def checkpoint_label(path: Path) -> str:
    if path.name == "pretrained_model" and path.parent.name:
        return path.parent.name
    return path.name


def reset_pipeline(obj: Any) -> None:
    reset = getattr(obj, "reset", None)
    if callable(reset):
        reset()


def to_batched_tensor(value: Any, *, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.float()
    else:
        tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim in (1, 3):
        tensor = tensor.unsqueeze(0)
    return tensor.to(device)


def make_observation(item: dict[str, Any], image_keys: list[str], device: torch.device) -> dict[str, torch.Tensor]:
    obs = {STATE_KEY: to_batched_tensor(item[STATE_KEY], device=device)}
    for key in image_keys:
        if key not in item:
            raise KeyError(f"dataset frame is missing required image key: {key}")
        obs[key] = to_batched_tensor(item[key], device=device)
    return obs


def validate_limits(action: np.ndarray) -> bool:
    for idx, (lower, upper) in enumerate(PIPER_LIMITS):
        value = float(action[idx])
        if lower is not None and value < lower:
            return False
        if upper is not None and value > upper:
            return False
    return True


def load_dataset(dataset_root: str, dataset_repo_id: str, dataset_revision: str):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(dataset_repo_id, root=dataset_root, revision=dataset_revision, tolerance_s=0.5)


def episode_frame_indices(dataset, episode_index: int) -> list[int]:
    ep = np.asarray(dataset.hf_dataset["episode_index"])
    indices = np.where(ep == episode_index)[0]
    return [int(i) for i in indices]


def resolve_episode_indices(dataset, args: argparse.Namespace) -> list[int]:
    if args.episode_indices:
        return [int(x) for x in args.episode_indices.split(",") if x.strip()]
    all_eps = sorted({int(x) for x in np.asarray(dataset.hf_dataset["episode_index"])})
    count = len(all_eps) if args.episodes <= 0 else min(args.episodes, len(all_eps))
    return all_eps[-count:]


def load_policy(checkpoint: Path, device: torch.device, local_files_only: bool):
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    policy = ACTPolicy.from_pretrained(
        str(checkpoint),
        local_files_only=local_files_only,
    ).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={
            "device_processor": {"device": device.type},
            "normalizer_processor": {"device": device.type},
        },
        postprocessor_overrides={
            "unnormalizer_processor": {"device": device.type},
            "device_processor": {"device": "cpu"},
        },
    )
    return policy, preprocessor, postprocessor


def predict_one(
    policy,
    preprocessor,
    postprocessor,
    obs: dict[str, torch.Tensor],
    mode: str,
) -> np.ndarray:
    with torch.inference_mode():
        processed = preprocessor(obs)
        if mode == "checkpoint_selection":
            chunk = policy.predict_action_chunk(processed)
            chunk = postprocessor(chunk)
            action = chunk[:, 0, :]
        else:
            action = policy.select_action(processed)
            action = postprocessor(action)
    return action.squeeze(0).detach().cpu().numpy().astype(np.float64)


def evaluate_episode(
    *,
    dataset,
    checkpoint_name: str,
    episode_index: int,
    max_frames: int,
    mode: str,
    policy,
    preprocessor,
    postprocessor,
    image_keys: list[str],
    device: torch.device,
) -> EpisodeStats:
    frame_indices = episode_frame_indices(dataset, episode_index)
    if max_frames > 0:
        frame_indices = frame_indices[:max_frames]

    preds: list[np.ndarray] = []
    gts: list[np.ndarray] = []
    errors = 0
    nan_count = 0
    limit_warnings = 0

    reset_pipeline(policy)
    reset_pipeline(preprocessor)
    reset_pipeline(postprocessor)

    for frame_index in frame_indices:
        item = dataset[frame_index]
        if mode == "checkpoint_selection":
            reset_pipeline(policy)
            reset_pipeline(preprocessor)
            reset_pipeline(postprocessor)
        try:
            obs = make_observation(item, image_keys, device)
            pred = predict_one(policy, preprocessor, postprocessor, obs, mode)
            gt = np.asarray(item[ACTION_KEY], dtype=np.float64).reshape(-1)[:7]
        except Exception as exc:  # noqa: BLE001
            errors += 1
            if errors <= 3:
                print(f"  ep{episode_index} frame{frame_index}: ERROR {type(exc).__name__}: {exc}")
            continue

        if pred.shape != (7,):
            errors += 1
            if errors <= 3:
                print(f"  ep{episode_index} frame{frame_index}: ERROR action shape {pred.shape}")
            continue
        if any(not math.isfinite(float(v)) for v in pred):
            nan_count += 1
        if not validate_limits(pred):
            limit_warnings += 1
        preds.append(pred)
        gts.append(gt)

    if preds:
        pred_arr = np.stack(preds)
        gt_arr = np.stack(gts)
        absdiff = np.abs(pred_arr - gt_arr)
        per_joint = absdiff.mean(axis=0)
        deltas = np.diff(pred_arr, axis=0)
        max_delta_norm = float(np.linalg.norm(deltas, axis=1).max()) if len(deltas) else 0.0
        gripper_trend = float(pred_arr[-1, 6] - pred_arr[0, 6]) if len(pred_arr) > 1 else 0.0
        mae = float(absdiff.mean())
        pfmax = float(absdiff.mean(axis=1).max())
    else:
        per_joint = np.full(7, np.nan)
        max_delta_norm = float("nan")
        gripper_trend = float("nan")
        mae = float("nan")
        pfmax = float("nan")

    return EpisodeStats(
        checkpoint=checkpoint_name,
        episode_index=episode_index,
        mode=mode,
        valid_frames=len(preds),
        total_frames=len(frame_indices),
        errors=errors,
        nan_count=nan_count,
        limit_warnings=limit_warnings,
        mae=mae,
        per_frame_mae_max=pfmax,
        max_delta_norm=max_delta_norm,
        gripper_pred_trend=gripper_trend,
        per_joint_mae={name: float(per_joint[i]) for i, name in enumerate(JOINT_NAMES)},
    )


def save_csv(path: Path, rows: list[EpisodeStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "checkpoint",
            "episode_index",
            "mode",
            "valid_frames",
            "total_frames",
            "errors",
            "nan_count",
            "limit_warnings",
            "mae",
            "per_frame_mae_max",
            "max_delta_norm",
            "gripper_pred_trend",
            *[f"mae_{name}" for name in JOINT_NAMES],
        ])
        for row in rows:
            writer.writerow([
                row.checkpoint,
                row.episode_index,
                row.mode,
                row.valid_frames,
                row.total_frames,
                row.errors,
                row.nan_count,
                row.limit_warnings,
                f"{row.mae:.6f}",
                f"{row.per_frame_mae_max:.6f}",
                f"{row.max_delta_norm:.6f}",
                f"{row.gripper_pred_trend:.6f}",
                *[f"{row.per_joint_mae[name]:.6f}" for name in JOINT_NAMES],
            ])


def main() -> int:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print("official_act_eval=true")
    print(f"device={device}")
    print("no_hardware_access=true")

    dataset = load_dataset(args.dataset_root, args.dataset_repo_id, args.dataset_revision)
    episode_indices = resolve_episode_indices(dataset, args)
    print(f"dataset_root={args.dataset_root}")
    print(f"dataset_repo_id={args.dataset_repo_id}")
    print(f"dataset_revision={args.dataset_revision}")
    print(f"episodes={episode_indices}")
    print(f"mode={args.mode}")

    all_rows: list[EpisodeStats] = []
    for checkpoint_str in args.checkpoint:
        checkpoint = Path(checkpoint_str)
        if (checkpoint / "pretrained_model").is_dir():
            checkpoint = checkpoint / "pretrained_model"
        label = checkpoint_label(checkpoint)
        print(f"\n=== checkpoint {label}: {checkpoint} ===")
        policy, preprocessor, postprocessor = load_policy(checkpoint, device, args.local_files_only)
        if args.exec_horizon is not None:
            policy.config.n_action_steps = int(args.exec_horizon)
        image_keys = [
            key for key, feature in policy.config.input_features.items()
            if key.startswith(IMAGE_PREFIX)
        ]
        print(
            f"chunk_size={policy.config.chunk_size} "
            f"n_action_steps={policy.config.n_action_steps} "
            f"images={image_keys}"
        )
        for episode_index in episode_indices:
            row = evaluate_episode(
                dataset=dataset,
                checkpoint_name=label,
                episode_index=episode_index,
                max_frames=args.max_frames,
                mode=args.mode,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                image_keys=image_keys,
                device=device,
            )
            all_rows.append(row)
            print(
                f"ck={label} ep={episode_index} valid={row.valid_frames}/{row.total_frames} "
                f"mae={row.mae:.6f} pfmax={row.per_frame_mae_max:.6f} "
                f"limit_warn={row.limit_warnings} nan={row.nan_count} "
                f"grip_trend={row.gripper_pred_trend:.6f}"
            )

    print("\n=== summary ===")
    for checkpoint in sorted({row.checkpoint for row in all_rows}):
        rows = [row for row in all_rows if row.checkpoint == checkpoint]
        print(
            f"{checkpoint}: avg_mae={np.mean([row.mae for row in rows]):.6f} "
            f"max_mae={np.max([row.mae for row in rows]):.6f} "
            f"limit_warnings={sum(row.limit_warnings for row in rows)} "
            f"nan={sum(row.nan_count for row in rows)} "
            f"errors={sum(row.errors for row in rows)}"
        )

    out_path = Path(
        args.output_csv
        or f"outputs/eval/official_act_single_cube_40/{args.mode}.csv"
    )
    save_csv(out_path, all_rows)
    print(f"\nCSV saved to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
