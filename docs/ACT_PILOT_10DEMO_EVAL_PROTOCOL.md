# ACT Pilot 10-Demo Evaluation Protocol

## Checkpoint

```
outputs/train/act_pilot_10demo/checkpoints/005000/pretrained_model
```

Use the latest checkpoint unless a specific one is chosen based on loss curves.

## Before every staged run

Reset is a separate reset-only command. It exits after moving to the recorded
start; it does not run policy inference.

```bash
python3 inference/deploy.py \
  --policy-type act-full \
  --checkpt outputs/train/act_pilot_10demo/checkpoints/005000/pretrained_model \
  --reset-to-recorded-start \
  --open-gripper-on-start \
  --gripper-start-open-value 0.0995 \
  --allow-real-full-e2e
```

Then run the selected stage command below with `--enforce-start-reset`.

## Staged evaluation (DO NOT skip stages)

### Stage A: Approach only

Verify the policy can reach the bottle region. No close, no lift.

```bash
python3 inference/deploy.py \
  --policy-type act-full \
  --checkpt outputs/train/act_pilot_10demo/checkpoints/005000/pretrained_model \
  --test-mode full-e2e \
  --full-e2e-stop-after approach \
  --approach-steps 400 \
  --open-gripper-on-start \
  --enforce-start-reset \
  --act-full-chunk-exec target_reached \
  --act-full-target-tol 0.04 \
  --save-rollout --save-final-images \
  --hold-after-stop 8 --no-auto-return \
  --debug-actions --debug-policy-io \
  --allow-real-full-e2e
```

Pass criteria:
- [ ] Gripper aligned with bottle (visually check final images)
- [ ] J2 reaches expected approach range (typically > 1.5)
- [ ] No collision with table or environment
- [ ] Arm is stable at stop pose

### Stage B: Close only

Run approach + close. Verify gripper closes on the bottle.

```bash
python3 inference/deploy.py \
  --policy-type act-full \
  --checkpt outputs/train/act_pilot_10demo/checkpoints/005000/pretrained_model \
  --test-mode full-e2e \
  --full-e2e-stop-after close \
  --approach-steps 600 \
  --open-gripper-on-start \
  --enforce-start-reset \
  --act-full-chunk-exec target_reached \
  --act-full-target-tol 0.04 \
  --save-rollout --save-final-images \
  --hold-after-stop 8 --no-auto-return \
  --debug-actions --debug-policy-io \
  --allow-real-full-e2e
```

Pass criteria:
- [ ] Approach criteria from Stage A all met
- [ ] Bottle is between gripper fingers before close
- [ ] Gripper closes ON the bottle (not in air)
- [ ] Object remains held at stop pose (no drop)
- [ ] step_log.csv confirms grip transition from open → close
- [ ] No accidental reopen after close detected

### Stage C: Lift only

Only run if Stage B passes on at least 2 out of 3 attempts.

```bash
python3 inference/deploy.py \
  --policy-type act-full \
  --checkpt outputs/train/act_pilot_10demo/checkpoints/005000/pretrained_model \
  --test-mode full-e2e \
  --full-e2e-stop-after lift \
  --approach-steps 600 \
  --open-gripper-on-start \
  --enforce-start-reset \
  --act-full-chunk-exec target_reached \
  --act-full-target-tol 0.04 \
  --save-rollout --save-final-images \
  --hold-after-stop 8 --no-auto-return \
  --debug-actions --debug-policy-io \
  --allow-real-full-e2e
```

Pass criteria:
- [ ] Approach and close criteria all met
- [ ] Bottle leaves the table surface
- [ ] Bottle does not drop immediately after lift
- [ ] Robot remains stable during lift

### Stage D: Full trajectory

Only run if Stage C passes on at least 2 out of 3 attempts.

```bash
python3 inference/deploy.py \
  --policy-type act-full \
  --checkpt outputs/train/act_pilot_10demo/checkpoints/005000/pretrained_model \
  --test-mode full-e2e \
  --full-e2e-stop-after full \
  --approach-steps 800 \
  --open-gripper-on-start \
  --enforce-start-reset \
  --act-full-chunk-exec target_reached \
  --act-full-target-tol 0.04 \
  --save-rollout --save-final-images \
  --hold-after-stop 8 --no-auto-return \
  --debug-actions --debug-policy-io \
  --allow-real-full-e2e
```

Pass criteria:
- [ ] All previous stage criteria met
- [ ] Full trained trajectory completes without unsafe motion
- [ ] Bottle remains held through the intended lift/hold portion
- [ ] No collision throughout full trajectory
- [ ] No unexpected reopen/drop

## Multi-position testing

After Stage C passes for center position, test left and right:

- Place bottle ~2-3 cm left of center
- Place bottle ~2-3 cm right of center

The policy should generalize to ±2-3 cm shifts if training included left/right demos.

## Rollout output

All rollouts save to:

```
logs/rollouts/test_a_YYYYMMDD_HHMMSS/
├── wrist_0000.jpg ...
├── global_0000.jpg ...
├── step_log.csv
├── step_0000.npz ...
└── final/
    ├── wrist.jpg
    └── global.jpg
```

## Go/No-Go criteria

| Stage | Min passes | Proceed to next stage? |
|-------|-----------|----------------------|
| A | 2/3 | → B |
| B | 2/3 | → C |
| C | 2/3 | → D |
| D | 2/3 | → Done |

If any stage fails to reach min passes:
1. Check rollout images for alignment issues
2. Check step_log.csv for gripper/joint trajectory
3. Consider fine-tuning or collecting more demos
4. Do NOT proceed to next stage until current stage passes
