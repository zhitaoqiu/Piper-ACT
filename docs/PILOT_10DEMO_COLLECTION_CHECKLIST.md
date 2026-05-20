# Pilot 10-Demo Collection Checklist

## Dataset target

```
data/lerobot_dataset_piper_bottle_pilot_10demo/
```

## Demo distribution

| Episode label | LeRobot index | Intent | Bottle position |
|---------------|---------------|--------|-----------------|
| `episode_000_center` | 000 | center | Center of marked region |
| `episode_001_center` | 001 | center | Center of marked region |
| `episode_002_center` | 002 | center | Center of marked region |
| `episode_003_center` | 003 | center | Center of marked region |
| `episode_004_center` | 004 | center | Center of marked region |
| `episode_005_center` | 005 | center | Center of marked region |
| `episode_006_left` | 006 | left | Slightly left of center (~2-3 cm) |
| `episode_007_left` | 007 | left | Slightly left of center (~2-3 cm) |
| `episode_008_right` | 008 | right | Slightly right of center (~2-3 cm) |
| `episode_009_right` | 009 | right | Slightly right of center (~2-3 cm) |

LeRobot assigns episode indices automatically in recording order. Record in
the order above to preserve the mapping. If a demo fails, press `R` while
recording and repeat the same slot before moving on.

## Recording command

```bash
bash scripts/record_piper_pilot_10demo.sh
```

Internally calls:
```bash
python3 teleop/data_collector.py \
    --task-mode full_pick_place \
    --dataset-root data/lerobot_dataset_piper_bottle_pilot_10demo/ \
    --dataset-repo-id piper/pilot_10demo \
    --record-gripper-action true
```

## Before every demo

- [ ] Robot reset to standard start qpos
  - Expected: `[-0.07682, 0.00623, -0.00392, 0.00000, 0.33034, 0.02376, 0.09950]`
  - Use manual alignment, or run deploy reset-only separately before collection
- [ ] Gripper open at ~0.0995 m (0.099–0.100)
- [ ] Reset guard passes (J2≈0.006, grip≈0.0995)
- [ ] Wrist camera (RealSense) streaming, no dropped frames
- [ ] Global camera (USB SN0002) streaming, no dropped frames
- [ ] Bottle placed in marked region
  - Center demos: bottle centered under approach path
  - Left demos: bottle shifted ~2-3 cm left
  - Right demos: bottle shifted ~2-3 cm right
- [ ] Operator confirms scene is clean (no obstacles)

## Each demo must include

- [ ] Approach aligned to bottle (J2 increases, arm moves toward bottle)
- [ ] Gripper closes ON the bottle, not in air
- [ ] Bottle is successfully lifted off table
- [ ] No collision with table or environment
- [ ] No severe slip (bottle stays between fingers)
- [ ] No failed grasp (gripper closes but bottle remains on table)
- [ ] No half trajectory (recording must capture full approach→close→lift)
- [ ] No accidental reopen during lift (gripper stays closed until return)

## Reject demo if

- [ ] Bottle is not positioned correctly (wrong location for the intended demo)
- [ ] Gripper closes before reaching bottle (close in air)
- [ ] Gripper closes in air (arm not deep enough yet)
- [ ] Bottle falls before lift completes
- [ ] Camera stream drops during recording (check preview window)
- [ ] qpos/action recording is discontinuous (check terminal for warnings)
- [ ] Gripper value does not show open-to-close transition
  - Start: ~0.0995
  - Close: drops to 0.045–0.060
- [ ] qpos/action dimension is not 7
- [ ] qpos/action values contain NaN/Inf
- [ ] Episode is too short (< 100 frames at 30 Hz control rate)
- [ ] Arm hits joint limit or singularity

## To discard the current episode

Press `R` during recording to discard and restart the current episode.

## Recording log

| Episode | Intent | Pass/Fail | Notes |
|---------|--------|-----------|-------|
| 000     | center |           |       |
| 001     | center |           |       |
| 002     | center |           |       |
| 003     | center |           |       |
| 004     | center |           |       |
| 005     | center |           |       |
| 006     | left   |           |       |
| 007     | left   |           |       |
| 008     | right  |           |       |
| 009     | right  |           |       |

Fill notes with any anomalies (e.g., "slight slip on lift", "approached a bit shallow").

Only clean, successful demos belong in the pilot dataset. Failed demos are
discarded, not kept as negative examples.

## After all demos collected

Run dataset sanity check:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate piper_act
python3 scripts/check_pilot_dataset.py \
    --dataset data/lerobot_dataset_piper_bottle_pilot_10demo/
```

This validates every episode and prints a pass/fail summary.
Do not proceed to training if fewer than 8 episodes pass.

The training script also runs this check and exits before training if the
dataset fails.
