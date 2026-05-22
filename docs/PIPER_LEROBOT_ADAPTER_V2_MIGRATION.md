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
- reset before recording and replay
- dataset checks before training
- ACT chunk smoothing ideas later in deployment work

Local measurements remain authoritative:

- qpos/action order is `[j1, j2, j3, j4, j5, j6, gripper]`
- joints are radians
- gripper is meters, with open around `0.0995`
- strong grasp close is around `0.045` to `0.055`
- current single-camera key is `observation.images.global_rgb`
- do not mix in the wrist camera for the first adapter-v2 one-demo path

## Stage 1 surface

The first adapter-v2 code lives outside the successful deploy path:

- `adapter_v2/piper_bus.py`: SDK-backed rad/meter motor bus built on the
  locally validated Piper wrapper
- `adapter_v2/piper_follower.py`: `piper_follower_v2` LeRobot robot
- `adapter_v2/piper_leader.py`: optional `piper_leader_v2` teleoperator for a
  validated second Piper CAN path
- `adapter_v2/reset.py`: standard start reset, reset guard, and gripper open
- `scripts/record_adapter_v2.py`: adapter registration plus standard LeRobot
  record entrypoint
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
close value, and reopening. It uses torque-retaining disconnect behavior: a
gripper-only check must not disable an unsupported arm pose at exit.

### Step 3: reset to standard start

```bash
python3 scripts/adapter_v2_reset_to_start.py --can-port can0
```

The default start pose is seeded from the successful 10-demo baseline. Treat it
as adapter-v2 data only after local reset tolerance passes. Override it with
`--q-start j1,j2,j3,j4,j5,j6,gripper` when validating a new start pose.

### Step 4: standard record path

Record one single-camera demo only after Step 3 passes. Reset motion must happen
before `record`; it must not be inside the saved episode.

The standard two-arm software leader command shape is:

```bash
python3 scripts/record_adapter_v2.py \
  --robot.type=piper_follower_v2 \
  --robot.can_port=can0 \
  --robot.cameras="{global_rgb: {type: opencv, index_or_path: /dev/video6, width: 640, height: 480, fps: 30}}" \
  --teleop.type=piper_leader_v2 \
  --teleop.can_port=<validated-leader-can-port> \
  --dataset.repo_id=piper/adapter_v2_one_demo \
  --dataset.root=data/lerobot_dataset_piper_adapter_v2_one_demo \
  --dataset.num_episodes=1 \
  --dataset.single_task="Pick and place the fixed bottle" \
  --dataset.push_to_hub=false \
  --dataset.fps=30
```

Do not invent a leader CAN name. Validate the actual leader path first. If only
the current one-CAN mirror mode is available, keep collection on hold for this
standard-record stage until its action source is explicitly decided.

### Step 5: dataset sanity check

```bash
python3 scripts/check_pilot_dataset.py \
  --dataset data/lerobot_dataset_piper_adapter_v2_one_demo \
  --expected-episodes 1 \
  --min-pass-episodes 1 \
  --require-single-camera \
  --camera-key observation.images.global_rgb
```

Check FPS, state/action dims, frames, single camera key, gripper transition,
NaN/Inf, and start-pose consistency before training.

### Step 6: replay the recorded demo

Reset first, then replay the saved one-demo episode:

```bash
python3 scripts/adapter_v2_reset_to_start.py --can-port can0

python3 scripts/replay_adapter_v2.py \
  --robot.type=piper_follower_v2 \
  --robot.can_port=can0 \
  --dataset.repo_id=piper/adapter_v2_one_demo \
  --dataset.root=data/lerobot_dataset_piper_adapter_v2_one_demo \
  --dataset.episode=0
```

Do not train ACT until replay succeeds on the adapter-v2 data path.
