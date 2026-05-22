# Adapter v2 10-Demo Collection Report

Date: 2026-05-22
Branch: `piper-lerobot-adapter-v2`

## 1. Collection Configuration

| Setting | Value |
|---|---|
| Script | `scripts/record_adapter_v2_mirror.py` |
| CAN port | can0 |
| Global camera | /dev/video6 (5MP USB Camera) |
| Start guard mode | zone |
| Start pose file | `config/adapter_v2_start_pose.json` |
| FPS | 10 |

## 2. Dataset Summary

| Metric | Value |
|---|---|
| Dataset path | `data/lerobot_dataset_piper_bottle_adapter_v2_10demo/` |
| Episodes | 10 |
| Total frames | 1472 |
| Avg frames/episode | 147.2 (range: 140–165) |
| State dim | 7 (j1–j6, gripper) |
| Action dim | 7 (j1–j6, gripper) |
| Camera | `observation.images.global_rgb` (single) |
| FPS | 10 |
| Format | LeRobot v3 (parquet + video) |

## 3. Per-Episode Results

| Ep | Frames | Gripper min | Gripper first | Lift | Notes |
|---|---|---|---|---|---|
| 0 | 142 | 0.04530 | 0.09950 | PASS (0.85) | |
| 1 | 145 | 0.04810 | 0.09670 | PASS (1.04) | |
| 2 | 142 | 0.04850 | 0.09950 | PASS (0.67) | |
| 3 | 143 | 0.04810 | 0.09950 | PASS (0.96) | |
| 4 | 150 | 0.04810 | 0.09950 | PASS (1.07) | |
| 5 | 165 | 0.04810 | 0.09940 | PASS (0.89) | |
| 6 | 148 | 0.05070 | 0.09940 | PASS (1.09) | |
| 7 | 146 | 0.04720 | 0.09950 | PASS (0.75) | |
| 8 | 151 | 0.04690 | 0.09940 | PASS (0.89) | |
| 9 | 140 | 0.04440 | 0.09950 | PASS (0.70) | Gripper min 0.0006 below ideal [0.045, 0.060], well within strong close range (>= 0.035) |

All 10 episodes: PASS.

## 4. Quality Checks

| Check | Result |
|---|---|
| All episodes have valid state/action (7D) | PASS |
| All episodes have single camera images | PASS |
| All episodes show gripper transition (open→close) | PASS |
| All episodes show lift phase after close | PASS (range 0.67–1.09) |
| Frame counts within range [80, 1200] | PASS |
| Black frame detection | PASS (all frames non-black) |
| Start guard mode recorded in metadata | PASS (zone) |

## 5. Sanity Check Result

```
10 passed, 0 failed, 10 total
RESULT: PASS. Dataset is trainable (10 >= 10).
```

## 6. Training

Training script: `scripts/train_act_adapter_v2_10demo.sh`
Output: `outputs/train/act_adapter_v2_10demo/`

| Parameter | Value |
|---|---|
| Policy | ACT |
| chunk_size | 10 |
| n_action_steps | 10 |
| dim_model | 128 |
| dim_feedforward | 512 |
| n_heads | 4 |
| n_encoder_layers | 2 |
| n_decoder_layers | 2 |
| dropout | 0.0 |
| use_vae | false |
| optimizer_lr | 3e-4 |
| optimizer_lr_backbone | 1e-4 |
| batch_size | 8 |
| steps | 3000 |
| save_freq | 1000 |
| eval_freq | 1000 |
