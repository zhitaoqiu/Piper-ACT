# VA11Hall Piper LeRobot Column Report

Date: 2026-05-20

Primary source: VA11Hall Zhihu column, "Piper部署ACT模型": https://www.zhihu.com/column/c_1952861476349019656

Focused articles:

- "piper移植lerobot开发记录": https://zhuanlan.zhihu.com/p/1942596287334711514
- "piper sdk接口函数整理": https://zhuanlan.zhihu.com/p/1943706175024637040
- "piper移植lerobot方案优化": https://zhuanlan.zhihu.com/p/1945534774732125791
- "Piper移植lerobot运动控制优化": https://zhuanlan.zhihu.com/p/1946636125415401016

Related column context used:

- "LeRobot开发笔记": https://zhuanlan.zhihu.com/p/1939413955182322506
- "Piper ACT试验记录": https://zhuanlan.zhihu.com/p/1952860586288346406
- CSDN mirror of the motion-control article, useful because it exposes the same original Zhihu attribution: https://blog.csdn.net/2508_90533928/article/details/151217044

Local code compared:

- [hardware/piper_wrapper.py](/home/huatec/piper_act_bottle_grasp/hardware/piper_wrapper.py)
- [piper_sdk_py_driver/piper_sdk_py_driver/sdk_adapter.py](/home/huatec/piper_act_bottle_grasp/piper_sdk_py_driver/piper_sdk_py_driver/sdk_adapter.py)
- [teleop/data_collector.py](/home/huatec/piper_act_bottle_grasp/teleop/data_collector.py)
- [inference/deploy.py](/home/huatec/piper_act_bottle_grasp/inference/deploy.py)
- [expert/lerobot_piper3/src/lerobot/robots/piper_follower/piper_follower.py](/home/huatec/piper_act_bottle_grasp/expert/lerobot_piper3/src/lerobot/robots/piper_follower/piper_follower.py)
- [expert/lerobot_piper3/src/lerobot/motors/piper/piper.py](/home/huatec/piper_act_bottle_grasp/expert/lerobot_piper3/src/lerobot/motors/piper/piper.py)
- [expert/lerobot_piper3/src/lerobot/teleoperators/piper_leader/piper_leader.py](/home/huatec/piper_act_bottle_grasp/expert/lerobot_piper3/src/lerobot/teleoperators/piper_leader/piper_leader.py)

## Executive Summary

The column connects Piper to LeRobot by implementing normal LeRobot hardware abstractions: a `piper_follower` robot, a `piper_leader` teleoperator, and a Piper SDK based motorbus. The intent is to keep `record.py`, dataset writing, ACT training, and policy deployment close to the standard LeRobot flow. Piper-specific logic is concentrated in robot, teleop, motorbus, CAN setup, and optional ACT inference smoothing.

That structure is directly useful, but the code should not be copied blindly. The column/fork has three important conflicts with this project:

- It assumes two Piper arms connected through separate CAN interfaces for software leader-to-follower teleop. Our current project uses a working one-CAN hardware mirror setup on `can0`.
- It has a unit-conversion ambiguity: the leader action is converted to radians/meters, but the shown follower observation path can expose raw Piper joint units. Our adapter already converts state and action consistently to radians/meters.
- It comments gripper range as `0..0.08 m`, while our measured real start/open is about `0.0995 m` and useful strong close is about `0.045..0.060 m`.

Recommendation: keep the current adapter for the 10-demo pilot and borrow specific components and practices from the column.

## 1. What The Column Implementation Does

The column treats LeRobot as the main workflow shell and Piper as a custom hardware backend.

Core pieces:

- `piper_follower`: a LeRobot `Robot` implementation copied from a SO100/SO101 follower shape and rewritten internally for Piper.
- `piper_leader`: a LeRobot `Teleoperator` implementation that reads another Piper arm as the leader.
- `PiperMotorsBus`: a thin Piper SDK wrapper that exposes `connect`, `read`, and `write` methods compatible with the robot and teleop classes.
- Robot config and teleop config register `piper_follower` and `piper_leader` as LeRobot config subclasses.
- `record.py` and `teleoperate.py` are mostly left alone. The column says the main changes are importing/registering the Piper classes and removing irrelevant serial `port` fields because Piper uses CAN.
- LeRobot dataset collection uses `python -m lerobot.record` / `lerobot-record`, with `--robot.type=piper_follower`, `--teleop.type=piper_leader`, camera config, `--dataset.repo_id`, `--dataset.num_episodes`, and `--dataset.push_to_hub=false`.
- Training uses standard ACT through `lerobot-train --policy.type=act`.
- Deployment in the article's LeRobot version uses the record/eval path with a `--policy.path=.../pretrained_model` argument replacing teleop control.

The later ACT experiment note adds that data quality and camera viewpoint mattered more than just code changes: reduce light variation, keep useful wrist/side views, remove distracting table textures, make demonstrations slower and more consistent, and diversify object position.

## 2. Directly Useful Parts For This Project

Useful to borrow:

- The standard LeRobot abstraction boundary: keep robot, teleop, and motorbus hardware logic isolated.
- The `PiperMotorsBus` idea, but with our current cleaner unit conversion.
- Feature convention: one state/action vector ordered as six joints plus gripper.
- The explicit Piper SDK function audit: `ConnectPort`, `ModeCtrl`, `EnableArm`, `GripperCtrl`, `GetArmJointMsgs`, `isOk`, `JointConfig`, and `CrashProtectionConfig`.
- CAN setup discipline. The column found unstable connection came from sloppy CAN activation.
- Deployment smoothing idea: reduce discontinuities between ACT chunks with interpolation/filtering.
- Data collection discipline: better camera placement, slower demonstrations, cleaner lighting, no distracting mat, and enough successful demos.
- The warning from the later experiment: do not train with a casually patched "magic ACT" if channel/dimension assumptions are not validated.

## 3. Conflicts With Our Current Adapter

| Area | Column / fork | Current project | Assessment |
|---|---|---|---|
| CAN topology | Separate leader/follower CAN names such as `can_master1` / `can_slave1` or `can_follower` | One working hardware mirror bus: `can0` | Keep current for pilot. Two-CAN migration is not minimum work. |
| Teleop | Software leader reads `piper_leader.get_action()` and sends to follower | Hardware mirror mode; custom recorder reads follower state and cameras | Keep current recorder for pilot. |
| Robot key names | `joint_1.pos` ... `joint_6.pos`, `gripper.pos` | `j1.pos` ... `j6.pos`, `gripper.pos`; dataset names `j1..j6, gripper` | Do not rename for this pilot unless fully migrating datasets. |
| Joint units | Leader action converts raw `0.001 deg` to radians; follower observation path in shown code can remain raw | Adapter always converts SDK raw to radians for state and radians back to raw for commands | Current adapter is safer and more internally consistent. |
| Gripper range | Comments and examples use `0..0.08 m` | Measured open around `0.0995 m`; strong close around `0.045..0.060 m`; clamp max `0.101 m` | Use our measured scale. |
| Collision protection | Article disables all collision protection to prevent data-collection disconnects | Current workflow rejects collisions and emphasizes clean demos | Do not disable protection as a default pilot practice. |
| Disconnect behavior | Initial implementation returned home on disconnect, later removed | Our deploy has recorded-start reset guard and can leave arm enabled during reset-only | Keep recorded-start guard. |
| ACT smoothing | Patch `select_action` queue, add interpolation and mean filter | Custom deploy has EMA, per-step replanning, max step limits, target-reached chunk execution, wrist-freeze handling | Borrow concept; do not patch training path now. |

## 4. Does It Use Standard LeRobot Record/Train/Eval?

Mostly yes.

The column's main design is standard LeRobot record/train/deploy with custom hardware classes. It does not rewrite the dataset format or ACT training loop for the base route. The required code changes are registration/import of `piper_follower` and `piper_leader`, plus the Piper motorbus/config classes.

There are two caveats:

- The motion-control article proposes modifying ACT `select_action` for inference smoothing and optionally adding a smoothness loss. That is a fork-level ACT patch, not standard LeRobot behavior.
- Newer LeRobot versions have evolved. In our local LeRobot tree, `lerobot_record.py` describes itself as a pure teleoperation recorder and points policy deployment toward rollout tooling. The column was written against a LeRobot version where record with `--policy.path` was used for deployment.

## 5. Piper Qpos / Action Representation

Column/fork representation:

- Order: `joint_1`, `joint_2`, `joint_3`, `joint_4`, `joint_5`, `joint_6`, `gripper`.
- LeRobot feature keys: `{motor}.pos`, producing `joint_1.pos` ... `gripper.pos`.
- Leader action conversion: arm raw Piper joint angles divided by about `57324.840764` to radians; gripper raw divided by `1_000_000` to meters.
- Write path expects six arm targets in radians and gripper in meters, then converts to Piper raw units before `JointCtrl` / `GripperCtrl`.
- Follower observation in the article/fork is ambiguous because the shown `read()` returns raw joint state values, while `get_observation()` stores them directly. If used exactly that way, state and action units may differ.

Our representation:

- Order: `[j1, j2, j3, j4, j5, j6, gripper]`.
- Dataset features: `observation.state` shape `(7,)`, `action` shape `(7,)`, names `["j1", ..., "gripper"]`.
- Units: arm radians, gripper meters for both qpos and action.
- Action in our custom recorder is next observed state, so it is an absolute target in the same unit/order as qpos.

## 6. Gripper Handling

Column/fork:

- SDK command unit is raw `0.001 mm`.
- LeRobot action gripper is intended to be meters.
- `GripperCtrl(abs(gripper_range), 1000, 0x01, 0)` sends position and effort.
- Comments cite `0..0.08 m`.
- SDK article notes `GripperCtrl` can clear errors and set zero.

Our project:

- SDK raw is converted with `raw / 1e6`.
- Command is clipped to `0..0.101 m`.
- Empirical open/start is about `0.0995 m`.
- Strong object close is about `0.045..0.055 m`, with pilot acceptance around `0.045..0.060 m`.
- Deployment already has open-on-start and relative close detection.

Conclusion: keep our gripper scale. The column confirms the raw-to-meter conversion but not our actual open limit.

## 7. Gripper Scale Comparison

| Value | Column | Current measured project |
|---|---:|---:|
| Open / max used | `0.08 m` in comments | about `0.0995 m` start/open |
| Raw conversion | meters `* 1e6` | meters `* 1e6` |
| Strong close | not quantified in article | about `0.045..0.060 m` |
| Command effort | example `1000` | default `1000` |

The conversion matches. The physical scale differs. The pilot dataset checker should use the measured project scale, not the article's `0.08`.

## 8. Camera Keys

Column:

- Early LeRobot examples use `top` and `side` camera config names.
- Later Piper ACT trials add a wrist/end-effector RealSense camera.
- The article emphasizes that camera names must match between recording and deployment.
- The final data-quality note says camera viewpoint was critical: a right-upper view closer to the demonstrator's view, a lower side view for gripper height, and wrist view were more useful than a poorly placed top/side pair.

Current project:

- Dataset keys are `observation.images.wrist_rgb` and `observation.images.global_rgb`.
- Wrist is RealSense. Global is a USB camera.
- These names are already used by training and deployment scripts.

Conclusion: do not rename camera keys for the 10-demo pilot. Improve placement and lighting instead.

## 9. ACT Chunk Smoothing / Interpolation

The motion-control article identifies a real ACT deployment issue: when `select_action` consumes an action queue and predicts a new chunk only after the queue is empty, the previous chunk tail and next chunk head may not be continuous.

Proposed fix:

- Save the last action when the old queue has one item left.
- When the next chunk is generated, linearly interpolate from the saved last action into the new chunk.
- Apply a moving-average filter over the new action sequence.
- Optionally add a smoothness loss over predicted actions during training.

The later ACT experiment warns not to train with an unvalidated modified ACT path because it caused channel-count errors in their setup.

Our project already has deployment-side controls:

- `--replan-every-step`
- EMA `--action-smooth`
- per-joint max delta clamps
- `target_reached` chunk execution
- wrist freeze / target-reached conflict handling
- relative close detection
- rollout logging and image saving

Recommendation for pilot: keep training standard ACT; keep smoothing in deployment/eval, and only add a separate tested chunk-boundary interpolation later if target-reached plus EMA is not enough.

## 10. Motion Discontinuity Avoidance

Column method:

- Smooth ACT chunks by interpolation and mean filtering.
- Consider smoother demonstrations during data collection.
- Avoid overlong predict/execute settings that can make the robot hesitate.

Our method:

- Move to recorded start before rollout.
- Enforce start guard.
- Clip arm deltas.
- Smooth targets with EMA.
- Execute chunk steps only after target is reached, with max-hold fallback.
- Ignore frozen wrist joints in target-reached checks when wrist freeze is active.

For the 10-demo pilot, our method is more safety-aware. Borrow the column's chunk-boundary interpolation as a future optional refinement, not a precondition for collection.

## 11. Piper Follower / Motorbus Abstraction

Yes. The column uses a Piper follower and a motorbus-like abstraction.

Expected classes:

- `PiperFollowerConfig`
- `PiperFollower`
- `PiperLeaderConfig`
- `PiperLeader`
- `PiperMotorsBusConfig`
- `PiperMotorsBus`

The checked-in fork under [expert/lerobot_piper3](/home/huatec/piper_act_bottle_grasp/expert/lerobot_piper3/src/lerobot) contains these classes and closely matches the column.

## 12. Minimum Changes Needed To Collect 10 Clean ACT Demos

Minimum for this repo:

- Keep current one-CAN mirror-mode collection through [teleop/data_collector.py](/home/huatec/piper_act_bottle_grasp/teleop/data_collector.py).
- Record to `data/lerobot_dataset_piper_bottle_pilot_10demo/`.
- Use full successful demonstrations, not approach-only snippets.
- Keep qpos/action order `[j1..j6, gripper]` in radians/meters.
- Keep camera keys `observation.images.wrist_rgb` and `observation.images.global_rgb`.
- Start each demo from the same recorded/standard start.
- Confirm gripper open around `0.0995 m` before each demo.
- Reject failed grasps immediately; do not train on them.
- Run `scripts/check_pilot_dataset.py` and require at least 8 passing episodes before any training.
- Use ACT only. Do not switch to SmolVLA, Diffusion Policy, or OpenVLA.

## 13. Full Migration Or Borrow Components?

Do not fully migrate before the 10-demo pilot.

Full migration would require:

- Reworking from one-CAN hardware mirror mode to two-CAN software teleop, or writing a standard LeRobot teleop abstraction for the current mirror mode.
- Renaming state/action feature keys or rebuilding dataset conventions.
- Validating state/action unit conversion in the fork.
- Validating camera integration under the current local LeRobot version.
- Retesting gripper scale and start/reset behavior.

That is too much surface area before a small pilot collection.

## 14. What To Keep From Current Implementation

Keep:

- Piper SDK adapter with explicit radian/meter conversion.
- One-CAN `can0` hardware mirror workflow for pilot collection.
- Current dataset feature order `[j1..j6, gripper]`.
- Camera keys `wrist_rgb` and `global_rgb`.
- Gripper open-on-start and measured scale around `0.0995`.
- Recorded-start reset and reset guard.
- ACT full checkpoint loading and processor handling.
- Target-reached chunk execution.
- Wrist freeze handling.
- Relative close detection.
- Rollout logging and image saving.
- Dataset/debug scripts.

## 15. What To Deprecate From The Single-Demo Overfit Path

Deprecate as mainline work:

- Further hard tuning of `outputs/train/act_full_fixed_overfit/`.
- Full real-robot task attempts from the old single-demo checkpoint as the main objective.
- One-off gripper threshold tuning for the old checkpoint.
- Treating code-forced close/lift as the target solution.
- Diffusion/SmolVLA/OpenVLA side routes for this objective.

Do not delete:

- Old datasets.
- Old checkpoints.
- Reset guard, debug rollout, gripper analysis, and safety utilities produced during the single-demo phase.

Those pieces remain useful as diagnostics and safety scaffolding.

## Concrete Recommendation

Keep current adapter but borrow specific components.

Use the VA11Hall column as the architecture reference for where Piper should sit in the LeRobot ecosystem, but do not replace the working current adapter before the pilot. Borrow the standard robot/teleop/motorbus separation, CAN setup discipline, camera/data-quality lessons, and deployment smoothing ideas. For the immediate objective, collect the 10 clean demos with the current mirror-mode recorder, validate them with the pilot checker, train ACT only after at least 8 episodes pass, and evaluate in staged A/B/C/D order.
