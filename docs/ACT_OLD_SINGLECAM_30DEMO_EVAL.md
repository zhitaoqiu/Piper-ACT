# ACT Old Single-Camera 30-Demo Staged Evaluation

Checkpoint: `outputs/train/act_old_singlecam_30demo/checkpoints/last/pretrained_model/`
Dataset: `data/lerobot_dataset_piper_bottle_old_singlecam_30demo_clean/`
Camera: `observation.images.global_rgb` only — single-camera, no wrist.

Do not run the robot automatically. Each stage must be verified by the operator before proceeding to the next.

---

## Reset Before Each Stage

```bash
python3 inference/deploy.py \
  --policy-type act-full \
  --checkpt outputs/train/act_old_singlecam_30demo/checkpoints/last/pretrained_model \
  --reset-to-recorded-start \
  --open-gripper-on-start \
  --no-wrist --global-camera auto \
  --allow-real-full-e2e
```

---

## Stage A: Approach

```bash
python3 inference/deploy.py \
  --policy-type act-full \
  --checkpt outputs/train/act_old_singlecam_30demo/checkpoints/last/pretrained_model \
  --test-mode full-e2e --full-e2e-stop-after approach \
  --hz 10 --approach-steps 220 \
  --no-wrist --global-camera auto \
  --open-gripper-on-start --enforce-start-reset \
  --act-full-chunk-exec target_reached --act-full-target-tol 0.04 \
  --save-rollout --save-final-images \
  --hold-after-stop 8 --no-auto-return \
  --debug-actions --debug-policy-io --allow-real-full-e2e
```

Pass criteria:

- [ ] Robot resets to recorded start pose without error
- [ ] Gripper opens to ~0.0995 before policy rollout
- [ ] Arm approaches the bottle position (not air, not table edge)
- [ ] End-effector height is reasonable (not too high, not crashing into table)
- [ ] No joint limit or CAN bus error
- [ ] Rollout saved under `logs/rollouts/`

---

## Stage B: Close / Strong Close

Only run after Stage A passes.

```bash
python3 inference/deploy.py \
  --policy-type act-full \
  --checkpt outputs/train/act_old_singlecam_30demo/checkpoints/last/pretrained_model \
  --test-mode full-e2e --full-e2e-stop-after close \
  --hz 10 --approach-steps 260 \
  --no-wrist --global-camera auto \
  --open-gripper-on-start --enforce-start-reset \
  --act-full-chunk-exec target_reached --act-full-target-tol 0.04 \
  --save-rollout --save-final-images \
  --hold-after-stop 8 --no-auto-return \
  --debug-actions --debug-policy-io --allow-real-full-e2e
```

Pass criteria:

- [ ] Stage A pass criteria still hold
- [ ] Bottle is between fingers before gripper closes
- [ ] Gripper closes on the bottle, not in empty air
- [ ] Gripper does not reopen prematurely (strong close)
- [ ] Object remains held at the stop pose
- [ ] No collision or jerk during close phase

---

## Stage C: Lift

Only run after Stage B passes.

```bash
python3 inference/deploy.py \
  --policy-type act-full \
  --checkpt outputs/train/act_old_singlecam_30demo/checkpoints/last/pretrained_model \
  --test-mode full-e2e --full-e2e-stop-after lift \
  --hz 10 --approach-steps 300 \
  --no-wrist --global-camera auto \
  --open-gripper-on-start --enforce-start-reset \
  --act-full-chunk-exec target_reached --act-full-target-tol 0.04 \
  --save-rollout --save-final-images \
  --hold-after-stop 8 --no-auto-return \
  --debug-actions --debug-policy-io --allow-real-full-e2e
```

Pass criteria:

- [ ] Stage A and B pass criteria still hold
- [ ] Bottle visibly leaves the table surface
- [ ] Bottle does not drop immediately after lift
- [ ] Robot arm remains stable, no shaking or oscillation
- [ ] No safety stop triggered

---

## Stage D: Release

Only run after Stage C passes.

```bash
python3 inference/deploy.py \
  --policy-type act-full \
  --checkpt outputs/train/act_old_singlecam_30demo/checkpoints/last/pretrained_model \
  --test-mode full-e2e --full-e2e-stop-after release \
  --release-stop-min-steps 10 --hz 10 --approach-steps 360 \
  --no-wrist --global-camera auto \
  --open-gripper-on-start --enforce-start-reset \
  --act-full-chunk-exec target_reached --act-full-target-tol 0.04 \
  --save-rollout --save-final-images \
  --hold-after-stop 8 --no-auto-return \
  --debug-actions --debug-policy-io --allow-real-full-e2e
```

Pass criteria:

- [ ] Stage A, B, and C pass criteria still hold
- [ ] Policy reaches release onset after confirmed strong close
- [ ] Gripper visibly reopens enough to release the bottle
- [ ] Robot remains stable at the stop pose after release
- [ ] Bottle is placed (not dropped from height)

---

## Full Attempt

Only run after Stage A, B, C, and D all pass.

```bash
python3 inference/deploy.py \
  --policy-type act-full \
  --checkpt outputs/train/act_old_singlecam_30demo/checkpoints/last/pretrained_model \
  --test-mode full-e2e --full-e2e-stop-after full \
  --hz 10 --approach-steps 360 \
  --no-wrist --global-camera auto \
  --open-gripper-on-start --enforce-start-reset \
  --act-full-chunk-exec target_reached --act-full-target-tol 0.04 \
  --save-rollout --save-global-video --save-final-images \
  --hold-after-stop 8 --no-auto-return \
  --debug-actions --debug-policy-io --allow-real-full-e2e
```

Pass criteria:

- [ ] All staged pass criteria hold
- [ ] Full pick-and-place cycle completes: approach → close → lift → release
- [ ] Bottle is successfully grasped and released at destination
- [ ] No safety stop or error during the full sequence

---

## Troubleshooting

If a staged run fails, check before changing model family:

1. Camera placement — does the global camera view match the old dataset view?
2. Lighting — is the scene lit similarly to the training data?
3. Bottle position — is it at the same fixed position as in the 30 demos?
4. Start pose — does `--enforce-start-reset` place the arm at the correct recorded start?
5. Compare with 10-demo baseline results for the same stage.

## Comparison Baseline

The 10-demo baseline for reference:

```text
outputs/train/act_old_singlecam_10demo/checkpoints/003000/pretrained_model/
```

Do not overwrite either checkpoint. Keep both for comparison.
