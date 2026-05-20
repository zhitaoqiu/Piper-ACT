#!/usr/bin/env python3
"""Replay a collected episode as a side-by-side video for documentation."""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def replay_episode(dataset_root: str, episode: int, output: str):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("piper/bottle_grasp", root=dataset_root)
    eps = np.array(ds.hf_dataset["episode_index"])
    indices = np.where(eps == episode)[0]

    if len(indices) == 0:
        raise SystemExit(f"Episode {episode} not found")

    print(f"Episode {episode}: {len(indices)} frames")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = ds.fps
    out = None

    joint_names = ["J1", "J2", "J3", "J4", "J5", "J6", "Gripper"]

    for frame_idx, ds_idx in enumerate(indices):
        item = ds[int(ds_idx)]
        wrist_t = item["observation.images.wrist_rgb"]
        wrist = (wrist_t.numpy() if hasattr(wrist_t, 'numpy') else np.asarray(wrist_t)).astype(np.uint8)
        has_global = "observation.images.global_rgb" in item
        if has_global:
            gt = item["observation.images.global_rgb"]
            global_img = (gt.numpy() if hasattr(gt, 'numpy') else np.asarray(gt)).astype(np.uint8)
        state = np.asarray(item["observation.state"], dtype=np.float32)

        # Convert from (C,H,W) to (H,W,C) BGR for cv2
        if wrist.shape[0] == 3:
            wrist = wrist.transpose(1, 2, 0)
        wrist_bgr = cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR)
        wrist_bgr = np.ascontiguousarray(wrist_bgr)

        if has_global:
            if global_img.shape[0] == 3:
                global_img = global_img.transpose(1, 2, 0)
            global_bgr = cv2.cvtColor(global_img, cv2.COLOR_RGB2BGR)
            global_bgr = np.ascontiguousarray(global_bgr)
            global_bgr = cv2.resize(global_bgr, (wrist_bgr.shape[1], wrist_bgr.shape[0]))
            frame = np.ascontiguousarray(np.hstack([wrist_bgr, global_bgr]))
        else:
            frame = wrist_bgr.copy()

        h, w = frame.shape[:2]

        # Joint state overlay
        y0 = h - 110
        cv2.rectangle(frame, (0, y0), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, f"Frame {frame_idx+1}/{len(indices)}  Phase: {state[7]:.2f}",
                    (10, y0 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        for j in range(7):
            name = joint_names[j]
            val = state[j]
            color = (0, 255, 0) if abs(val) < 0.01 else (0, 255, 255) if abs(val) < 3.0 else (0, 0, 255)
            text = f"{name}: {val:+.3f}"
            col = j % 4
            row = j // 4
            x = 10 + col * 160
            y_text = y0 + 50 + row * 22
            cv2.putText(frame, text, (x, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        if out is None:
            out = cv2.VideoWriter(output, fourcc, fps, (w, h))

        out.write(frame)

    out.release()
    print(f"Saved {output} ({len(indices)} frames @ {fps}fps)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="data/lerobot_dataset_delta_phase")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output", default="docs/episode_replay.mp4")
    args = parser.parse_args()
    replay_episode(args.dataset_root, args.episode, args.output)


if __name__ == "__main__":
    main()
