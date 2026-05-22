# Piper LeRobot Adapter V2 Migration

Date: 2026-05-22

## Frozen baseline

Do not mutate the successful old single-camera ACT baseline while adapter v2 is
being built.

- Success commit: `8149206`
- Success branch before migration: `act-pilot-10demo`
- Migration branch: `piper-lerobot-adapter-v2`
- Freeze tag: `act-10demo-success-before-adapter-v2`
- Checkpoint:
  `outputs/train/act_old_singlecam_10demo/checkpoints/003000/pretrained_model/`
- Backup checkpoint tar:
  `frozen_success/act_old_singlecam_10demo_success_checkpoint.tar.gz`
- Backup rollout tar, including saved global videos:
  `frozen_success/act_10demo_success_rollouts.tar.gz`

The baseline result is no longer only an offline finding. The old single-camera
10-demo ACT route succeeded on the real Piper arm.

## Migration posture

VA11Hall is the primary adapter-v2 architecture template:

- Piper follower / leader separation
- Piper SDK-backed motor bus
- LeRobot robot and teleoperator abstractions
- standard LeRobot `record` and `replay` data path
- manual start alignment before recording and replay
- dataset checks before training
- ACT chunk smoothing ideas later in deployment work

Local measurements remain authoritative:

- qpos/action order is `[j1, j2, j3, j4, j5, j6, gripper]`
- joints are radians
- gripper is meters, with open around `0.0995`
- strong grasp close is around `0.045` to `0.055`
- current single-camera key is `observation.images.global_rgb`
- do not mix in the wrist camera for the first adapter-v2 one-demo path

## Current Piper topology

The local Piper teaching setup is a powered hardware mirror path:

- the teaching arm controls the follower directly after the teaching arm is
  powered
- the computer sees the follower on `can0`
- the recorder reads follower qpos/gripper plus the global camera
- the recorded mirror action remains follower next-state action in the existing
  LeRobot dataset schema

There is no second leader CAN interface in the current setup. Adapter v2 must
not block the data path on a software `PiperLeader` teleoperator that this
hardware topology does not use.

There is no automatic reset motion in this teaching flow. The guarded recorder
checks the start pose before every episode. The operator may place the
teaching/follower pair near the start pose manually and press `C` to recheck, or
press `R` to request `reset_to_standard_start` outside recording. The `R` path
requires terminal confirmation and must only be used when the teaching hardware
state is safe for a commanded follower reset.

## Stage 1 surface

The first adapter-v2 code lives outside the successful deploy path:

- `adapter_v2/piper_bus.py`: SDK-backed rad/meter motor bus built on the
  locally validated Piper wrapper
- `adapter_v2/piper_follower.py`: `piper_follower_v2` LeRobot robot
- `adapter_v2/piper_leader.py`: retained VA11Hall-style software leader
  scaffolding, not the active local teaching path
- `adapter_v2/start_pose.py`: start-pose guard helpers with no motion
- `adapter_v2/reset.py`: explicit confirmed reset helper for the recorder `R`
  key outside recording
- `scripts/adapter_v2_check_start_pose.py`: read-only start-pose comparison
- `scripts/record_adapter_v2.py`: current one-CAN guarded recorder that writes
  LeRobotDataset episodes after a per-episode start guard
- `scripts/record_adapter_v2_mirror.py`: compatibility alias that forwards to
  the guarded Stage 1 recorder
- `scripts/replay_adapter_v2.py`: adapter registration plus standard LeRobot
  replay entrypoint

The existing ACT baseline deployment scripts are not the adapter-v2 target for
this stage.

## Validation order

Activate the environment that has `lerobot` and Piper SDK access before these
commands.

### Step 1: read-only smoke

```bash
python3 scripts/adapter_v2_smoke_read.py --can-port can0
```

Pass only when connect, qpos read, gripper read, finite values, and state
dimension 7 all pass.

### Step 2: operator-confirmed gripper test

```bash
python3 scripts/adapter_v2_gripper_test.py --can-port can0
```

This script asks for explicit confirmation before opening, moving to the safe
close value, and reopening. The motion is sent as a small-step sweep with a hold
at each end so it matches the locally successful close rhythm better than a
single gripper command. It uses torque-retaining disconnect behavior: a
gripper-only check must not disable an unsupported arm pose at exit.

For an empty gripper and an obvious close/open sweep during diagnosis:

```bash
python3 scripts/adapter_v2_gripper_test.py \
  --can-port can0 \
  --safe-close 0.000 \
  --settle-s 1.0
```

### Step 3: manually align the start pose

```bash
python3 scripts/adapter_v2_check_start_pose.py --can-port can0
```

This command sends no motion. Place the teaching/follower pair near the start
pose by hand, run the read-only check, adjust manually if it fails, and rerun.
The default comparison pose is seeded from the successful 10-demo baseline.
Override it with `--q-start j1,j2,j3,j4,j5,j6,gripper` only when validating a
new manual start target.

### Step 4: record one demo

Run the dry-run first. It connects Piper, prints the qpos start guard, validates
the key flow, writes no dataset, and sends no reset motion:

```bash
python3 scripts/record_adapter_v2.py \
  --can-port can0 \
  --dry-run
```

Record one single-camera demo only after the guard flow is clear:

```bash
python3 scripts/record_adapter_v2.py \
  --can-port can0 \
  --global-camera auto \
  --num-episodes 1 \
  --fps 10
```

The current machine inventory on 2026-05-22 shows only the working `can0`
interface because the powered teaching arm drives the follower directly. The
guarded recorder records that one-CAN hardware mirror path. It records only
`observation.images.global_rgb`, uses 7D `[j1..j6,gripper]` state/action, keeps
Piper enabled on exit, and saves adapter-v2 start metadata for every episode.
SPACE can start only after `START GUARD PASS`; SPACE or ENTER stops the episode.
Before any next episode, the recorder runs the start guard again.

`piper_leader_v2` remains reference scaffolding for a different software-leader
topology. Do not use it for the current powered teaching-arm recording flow.

### Step 5: dataset sanity check

```bash
python3 scripts/check_pilot_dataset.py \
  --dataset data/lerobot_dataset_piper_adapter_v2_one_demo \
  --expected-episodes 1 \
  --min-pass-episodes 1 \
  --require-single-camera \
  --camera-key observation.images.global_rgb \
  --expected-start-qpos 0.06292,0.00750,-0.00396,0.02732,0.30946,-0.09826,0.09950
```

Check FPS, state/action dims, frames, single camera key, gripper transition,
NaN/Inf, and start-pose consistency before training.

### Step 6: replay the recorded demo

Manually return near the episode start and check it before replay. Do not run an
automatic reset command in this topology:

```bash
python3 scripts/adapter_v2_check_start_pose.py --can-port can0

python3 scripts/replay_adapter_v2.py \
  --robot.type=piper_follower_v2 \
  --robot.can_port=can0 \
  --dataset.repo_id=piper/adapter_v2_one_demo \
  --dataset.root=data/lerobot_dataset_piper_adapter_v2_one_demo \
  --dataset.episode=0
```

Do not train ACT until replay succeeds on the adapter-v2 data path.
