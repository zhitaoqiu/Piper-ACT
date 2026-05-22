# VA11Hall Piper + LeRobot Lessons

## 0. Adapter V2 Strategy Update

As of 2026-05-22, the old single-camera 10-demo ACT baseline has succeeded on
the real Piper arm. VA11Hall is now the primary adapter-v2 architecture
template, not only a comparison reference. The migration lives on
`piper-lerobot-adapter-v2` and must leave the successful baseline checkpoint,
rollouts, videos, and deploy path available as the fallback demonstration.

The adapter-v2 migration should copy the follower/leader/motorbus and standard
record/replay discipline boldly, while revalidating local joint order, sign,
units, CAN names, gripper scale, camera devices, reset pose, motion limits, and
enable/disable behavior on this Piper setup.

The local teaching topology is now explicit: this Piper setup does not expose a
separate leader CAN to the recorder. After the teaching arm is powered, it
directly controls the follower arm. Adapter v2 records the follower through
`can0` with the global camera, so `PiperLeader` stays a VA11Hall-style reference
shape rather than the active Stage 1 data source.

参考：

- 知乎专栏：Piper部署ACT模型  
  https://www.zhihu.com/column/c_1952861476349019656
- 重点文章：
  - piper移植lerobot开发记录
  - piper sdk接口函数整理
  - piper移植lerobot方案优化
  - Piper移植lerobot运动控制优化

这份文档记录从 VA11Hall 专栏里抽出的工程经验，以及它们对当前 Piper bottle grasp 项目的实际意义。

## 1. Why this reference matters

这个专栏重要，是因为它不是泛泛讲 LeRobot，也不是只给一个改过名字的 fork，而是直接围绕 Piper 接入 LeRobot 工作流展开。它比随机 GitHub fork 更接近我们当前问题：怎样让 Piper 机械臂以 LeRobot 能理解的方式完成记录、训练和部署。

它最有价值的部分不只是 ACT 训练命令，而是 Piper-LeRobot adapter 的结构：

- Piper SDK wrapper
- Piper follower / robot interface
- `record.py` 集成方式
- qpos/action 格式
- gripper 读写和尺度处理
- camera key 管理
- ACT action chunk 的 smoothing / interpolation

对我们来说，这个专栏应当作为 Piper 接入 LeRobot 的架构参考，而不是直接照抄代码。

## 2. Main architecture learned from the column

专栏的主线是把 Piper 包装成 LeRobot 标准硬件对象：

- LeRobot 期望机器人暴露标准 robot interface。
- Piper 需要自定义 robot/follower 实现。
- Piper SDK 负责低层 CAN、关节、夹爪读写。
- adapter 负责把 Piper SDK 的状态和命令转换成 LeRobot 的 observation/action。
- 数据采集、训练、部署应尽量沿用 LeRobot 的 record/train/eval 流程。

这对当前项目的含义：

我们不应该为了“更像标准 LeRobot”立刻推翻现在能工作的安全 adapter。更合理的路线是保留当前 Piper CAN、reset guard、gripper scale、rollout logging 等安全层，然后逐步把代码组织成更清晰的 LeRobot-style robot abstraction。

## 3. Piper SDK and hardware interface lessons

Piper SDK 层需要重点关注：

- CAN 接口初始化和连接状态。
- 机械臂状态读取。
- 机械臂目标命令发送。
- 夹爪状态读取和命令发送。
- 连接不稳定、状态不同步、启动时姿态不可信等问题。
- startup/reset 检查不能省略。

对当前项目的含义：

- `open-gripper-on-start` 是必要的，不是临时补丁。
- recorded/standard start reset 是必要的。
- reset guard 是必要的。
- gripper scale 必须从真实数据里量出来，不能只相信注释或示例值。
- 部署前必须确认 CAN、相机、qpos/action 都在正常状态。

## 4. qpos/action and gripper scale lessons

Piper 的 qpos/action 顺序必须在 record、train、deploy 三处保持一致。单位转换也必须显式处理，尤其是：

- 关节 raw unit 和 radians 的转换。
- 夹爪 raw unit 和 meters 的转换。
- `observation.state` 和 `action` 是否使用同一尺度。
- 训练集里的 gripper 数值是否和部署命令一致。

错误的 gripper 尺度会让 policy 看起来像“模型坏了”，实际可能只是 open/close 数值错位。

我们当前测得的 gripper 经验值：

- gripper open 约为 `0.0995`
- 数据集里的强 close 约为 `0.045–0.055`
- sanity check 接受的 close 范围可放在约 `0.045–0.060`

close threshold 应该来自数据统计，而不是假设 open/close 中点。

## 5. Data collection lessons

专栏反复体现一个实际规律：早期少量干净成功 demo 比大量混杂 demo 更有价值。

数据采集要控制：

- start pose
- camera view
- gripper 初始 open 状态
- object position
- 光照和背景干扰
- 示教动作的一致性
- 是否存在失败抓取、碰撞、相机掉帧或 qpos/action 异常

对当前计划的含义：

- 旧的约 40 条单目 bottle grasp demo 已先筛成 10 条干净 demo。
- old-singlecam ACT baseline 必须只使用旧数据集的单目 camera key。
- 单目训练不能混双目部署。
- 当前 old-singlecam camera key 是 `observation.images.global_rgb`。
- 后续双目 pilot 数据集应作为独立路线，不和 old-singlecam baseline 混在一起。

## 6. Motion control and action smoothing lessons

ACT 的部署特性是输出 action chunks。chunk 内部动作可能平滑，但前后两个 chunk 的尾首不一定连续，所以真机上会出现跳变或抖动。

专栏里的优化思路：

- 记录上一段 action chunk 的最后一个 action。
- 新 chunk 生成后，用线性插值衔接旧 chunk 末尾和新 chunk 开头。
- 对 action sequence 做均值滤波。
- 需要考虑真机执行滞后和机械极限。
- 训练阶段可尝试 smoothness loss，但这属于额外改动，不能默认认为一定稳定。

我们自己的经验对应上了这一点：

- `target_reached` 用来处理 ACT chunk timing 和真机执行速度不匹配。
- wrist freeze 会影响 target_reached 判断，需要单独处理。
- action smoothing、max_delta、rollout logging 是部署层安全措施。
- chunk 不连续不一定是模型失败，很多时候是 deployment-layer 问题。

## 7. What we should keep from our current adapter

当前 adapter 里应该保留：

- Piper CAN control
- `open-gripper-on-start`
- recorded/standard start reset
- reset guard
- measured gripper scale
- dataset sanity checker
- rollout/image logging
- safe max_delta / smoothing
- staged evaluation：approach -> close -> lift

这些不是 ACT 专属资产。它们以后对 Diffusion 或其他 policy 也有用，因为它们解决的是真实 Piper 硬件、数据质量和部署安全问题。

## 8. What should not be carried forward blindly

不要把 single-demo overfit 阶段的调试逻辑变成长期架构：

- single-demo fixed-overfit `READY_J2` tuning
- single-demo `close_stop` logic
- overfitting-specific phase assumptions
- 把 ACT-only `target_reached` 当成所有模型通用执行方式
- 在 approach/close/lift 分阶段通过前直接跑 full-e2e

这些工具帮助我们查清连接、夹爪、reset、chunk execution、wrist freeze 等问题，但它们不应该变成未来所有模型和所有数据集的默认假设。

## 9. ACT vs Diffusion relevance

VA11Hall 专栏标题围绕 Piper 部署 ACT，但很多内容本质上是 Piper-LeRobot adapter 经验。

adapter-level lessons 对 ACT 和 Diffusion 都有用：

- Piper SDK wrapper
- robot/follower abstraction
- qpos/action format
- gripper scale
- camera keys
- dataset recording
- reset/start checks

ACT-specific lessons 主要是：

- ACT chunk execution
- action queue smoothing
- chunk discontinuity handling
- ACT deployment command structure

如果以后比较 Diffusion，应复用 common adapter/data layer，但不能直接复用 ACT-specific chunk logic。Diffusion 的动作时序和执行接口需要单独验证。

## 10. Current project decision

当前决策：

先用旧数据中筛出的 10 条单目 demo 训练 ACT baseline，验证固定位置 Piper bottle grasp 是否能跑通。

Dataset:

```text
data/lerobot_dataset_piper_bottle_old_singlecam_10demo/
```

Camera key:

```text
observation.images.global_rgb
```

Output:

```text
outputs/train/act_old_singlecam_10demo/
```

注意：

- 现在不要训练，等已有训练任务结束、GPU 空出来再跑。
- 不要把单目训练和双目部署混用。
- 在这个 ACT baseline 评估前，不切到 SmolVLA、Diffusion 或 OpenVLA。

## 11. Next steps

1. 等 GPU 空闲。
2. 运行 old-singlecam 10-demo ACT training。
3. 按阶段评估：
   - approach
   - close
   - lift
4. 如果固定位置 ACT baseline 成功，再考虑：
   - 采集 offset demos
   - 训练更强的 ACT 数据集
   - 可选地在同一份干净数据上比较 Diffusion

核心原则：先证明最小闭环，再扩大数据分布和模型复杂度。
