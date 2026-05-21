# ACT Old Single-Camera 30-Demo Run Report

Run date: 2026-05-21
Route: ACT old single-camera fixed-position Piper bottle baseline — 30 clean demos

## 1. Dataset path

```text
data/lerobot_dataset_piper_bottle_old_singlecam_30demo_clean/
```

## 2. Camera key

```text
observation.images.global_rgb
```

The trained policy input features are:

```text
observation.state
observation.images.global_rgb
```

No wrist camera is present in this checkpoint.

## 3. Dataset sanity result

Result: PASS — 30/30 episodes valid, single camera confirmed, no missing frames.

## 4. Training command

```bash
DEVICE=cuda bash scripts/train_act_old_singlecam_30demo.sh \
  > logs/train_act_old_singlecam_30demo_20260521_120011.log 2>&1
```

## 5. Training start/end time

- Start: 2026-05-21 12:00:11
- End: 2026-05-21 ~12:15

## 6. Training exit code

```text
0
```

## 7. Log path

```text
logs/train_act_old_singlecam_30demo_20260521_120011.log
```

## 8. Output directory

```text
outputs/train/act_old_singlecam_30demo/
```

## 9. Checkpoint path

Final checkpoint:

```text
outputs/train/act_old_singlecam_30demo/checkpoints/last/pretrained_model/
```

## 10. Whether CUDA was used

Yes — RTX 3060, ~5.4–5.7 steps/sec, 12M params.

## 11. Whether training finished successfully

Yes — 5000/5000 steps completed, no errors. All 5 checkpoints (1000/2000/3000/4000/5000) saved.

## 12. Offline evaluation result

| Checkpoint | Mean MSE |
|-----------|----------|
| 001000    | 0.003974 |
| 002000    | 0.001987 |
| 003000    | 0.000945 |
| 004000    | 0.000712 |
| 005000    | 0.000609 |

MSE decreases monotonically; no overfitting observed. Final MSE 0.000609.

## 13. Errors/warnings

None.

## 14. Next recommended step

Do not run the robot automatically.

Next manual evaluation should be staged per:
```text
docs/ACT_OLD_SINGLECAM_30DEMO_PLAN.md
```
