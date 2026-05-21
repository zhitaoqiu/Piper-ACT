# ACT Old Single-Camera: 10-Demo vs 30-Demo Comparison

## Dataset

| | 10-Demo | 30-Demo |
|---|---|---|
| Path | `data/lerobot_dataset_piper_bottle_old_singlecam_10demo/` | `data/lerobot_dataset_piper_bottle_old_singlecam_30demo_clean/` |
| Source | top 10 from 40 (manual) | top 30 from 40 (script, quality-scored) |
| Episodes | 10 | 30 |
| Total frames | 1793 | 5428 |
| FPS | 10 | 10 |
| Camera | `observation.images.global_rgb` | `observation.images.global_rgb` |
| Wrist camera | none | none |
| Selection method | manual pick | `scripts/select_old_singlecam_30demo_clean.py` |

## Training

| | 10-Demo | 30-Demo |
|---|---|---|
| Script | `scripts/train_act_old_singlecam_10demo.sh` | `scripts/train_act_old_singlecam_30demo.sh` |
| Steps | 3000 | 5000 |
| Duration | ~8.8 min | ~14.9 min |
| GPU | RTX 3060 | RTX 3060 |
| Speed | ~5.6 steps/s | ~5.6 steps/s |
| Params | 12.17M | 12.17M |
| Batch size | 8 | 8 |
| chunk_size | 10 | 10 |
| lr | 3e-4 | 3e-4 |
| Best checkpoint | step 3000 | step 5000 |

## Offline Evaluation

| | 10-Demo | 30-Demo |
|---|---|---|
| MSE (all) | 0.000654 | **0.000609** |
| MSE (arm) | 0.000762 | — |
| MSE (gripper) | 0.00000538 | — |
| MSE @ 1000 | — | 0.003974 |
| MSE @ 2000 | — | 0.001987 |
| MSE @ 3000 | 0.000654 | 0.000945 |
| MSE @ 4000 | — | 0.000712 |
| MSE @ 5000 | — | **0.000609** |
| Trend | converged at 3000 | monotonically decreasing through 5000 |

## Checkpoint

| | 10-Demo | 30-Demo |
|---|---|---|
| Path | `outputs/train/act_old_singlecam_10demo/checkpoints/003000/pretrained_model/` | `outputs/train/act_old_singlecam_30demo/checkpoints/last/pretrained_model/` |
| Size | ~48.7 MB | ~48.7 MB |
| Format | safetensors | safetensors |

## Expected Benefits of 30-Demo over 10-Demo

1. **3x more data** — 30 episodes vs 10 provides better coverage of grasp variations (approach angle, close timing, lift height)
2. **Quality-filtered** — top 30 selected from 40 by deterministic heuristics (grip open/close/release profile, frame count, data integrity), filtering out borderline demos
3. **Lower MSE** — 0.000609 vs 0.000654 (~7% improvement) at the final checkpoint
4. **Sustained convergence** — loss continues to decrease through 5000 steps without overfitting, suggesting the model benefits from the additional data
5. **Better generalization expected** — more diverse training trajectories should reduce the gap between training distribution and real-robot deployment

## Real-Robot Evaluation Plan

Both models follow the same staged evaluation protocol:

1. Stage A: Approach
2. Stage B: Close / Strong Close
3. Stage C: Lift
4. Stage D: Release
5. Full attempt (only after A/B/C/D pass)

Eval command references:
- 10-demo: [docs/ACT_OLD_SINGLECAM_10DEMO_EVAL.md](ACT_OLD_SINGLECAM_10DEMO_EVAL.md)
- 30-demo: [docs/ACT_OLD_SINGLECAM_30DEMO_EVAL.md](ACT_OLD_SINGLECAM_30DEMO_EVAL.md)

## Comparison Method

Run the same stage on both checkpoints back-to-back under identical conditions (same bottle position, same lighting, same camera placement). Compare:

- Which checkpoint reaches the bottle more reliably?
- Which produces a stronger close (gripper stays closed, no premature reopen)?
- Which lifts more stably (less shaking, fewer drops)?
- Which completes the full pick-and-place more often?

Do not delete or overwrite either checkpoint. Keep both for A/B comparison.
