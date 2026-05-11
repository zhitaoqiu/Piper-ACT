# 开发日记 — Piper ACT Bottle Grasp

## 2026-05-07 — 项目启动与环境搭建

### 设计决策

**选型：LeRobot + ACT**
- 决定基于 LeRobot 框架减少手写代码，利用其内置的 ACT 模型、数据集格式和训练 pipeline
- 选 conda 环境隔离，不用系统 Python 或 ROS2 环境

**选型：镜像模式 vs 软件转发**
- 最初设计了两条独立 CAN 总线（can0/can1），由上位机读示教臂、转发给被控臂
- 用户插电后发现：两只臂共享同一 CAN 总线，被控臂在硬件层面自动镜像示教臂
- 结论：不需要软件转发，数据采集只需读被控臂状态即可
- 教训：先确认硬件能力再设计软件方案

### 环境配置：PYTHONPATH 污染

**问题**：系统安装了 ROS2，`~/.bashrc` 中设置了 PYTHONPATH 指向 `/opt/ros/humble/lib/python3.10/site-packages`。即使激活 conda 环境，PYTHONPATH 优先级高于 conda，导致 import 时加载系统旧包。

**现象**：`import lerobot` 导入的是 `~/.local/` 下的旧版而非 conda 环境的最新版。

**解决方案**：
1. 在 `~/miniconda3/envs/piper_act/etc/conda/activate.d/unset_pythonpath.sh` 中 `unset PYTHONPATH`
2. 在 `deactivate.d/restore_pythonpath.sh` 中恢复
3. 启动命令前加 `PYTHONPATH=` 作为二次保险

### 依赖安装：国内镜像加速

**问题**：PyTorch 等大包从官方 PyPI 下载超时。

**尝试**：
- 清华源（Tsinghua）：部分包 403 禁止访问
- 中科大源（USTC）：可用，速度尚可
- 阿里云源：部分包同步滞后

**最终方案**：pip 使用 USTC 镜像，`pip install` 加 `-i https://mirrors.ustc.edu.cn/pypi/web/simple`

### NumPy ABI 冲突

**问题**：`import cv2` 报 `_ARRAY_API` 错误。

**原因**：OpenCV Python 包在编译时链接了 NumPy 1.x ABI，而 NumPy 2.x 不兼容。

**解决**：`pip install "numpy<2" --force-reinstall opencv-python`

### LeRobot 安装位置

**问题**：LeRobot 被意外安装到 `~/.local/`（系统级），conda 环境中找不到。

**解决**：`~/miniconda3/envs/piper_act/bin/python3 -m pip install -e ~/third_party/lerobot/` 直接安装到 conda 环境。同时删除 `~/.local/` 下的旧版本避免干扰。

---

## 2026-05-08 — 数据采集与训练

### 摄像头调试

**RealSense 深度回退**
- 初次启动时 `enable_depth=true` 导致 "Frame didn't arrive within 5000" 超时
- 分析：`config.enable_device(serial)` 指定序列号可能导致配置冲突
- 修复：去掉 `enable_device(serial)` 用自动检测，添加深度流异常的 try/catch 自动回退到 RGB-only

**USB 摄像头（SN0002）无法打开**
- `/dev/video0` 和 `/dev/video1`（SN0002）V4L2 打开失败
- 实现自动扫描：扫描所有 `/dev/video*` 设备，过滤掉 RealSense，逐个尝试
- 添加 `--list-cameras` 命令列出所有视频设备及 sysfs 名称
- 添加 `--global-camera auto` 自动选择，也支持 `--global-camera 6` 手动指定

**摄像头颜色问题**
- 用户反馈 USB 相机画面颜色偏绿
- RealSense BGR 格式与 OpenCV 显示兼容性检查——确认无需额外转换

### 数据采集功能迭代

**采集热键**
- R：丢弃当前录制中的 episode 并重录
- P/H 自动起点管理已移除：采集前由人工把示教臂和被控臂一起摆到固定起点

**LeRobot v3.0 格式兼容**
- 使用 `LeRobotDataset.create()` / `LeRobotDataset.resume()` 创建/续写数据集
- `create_or_resume_dataset()` 处理不完整数据集自动备份

**数据集验证**
- 20 episodes, 9,506 frames, ~5.3 min 总时长
- 平均 475 frames/episode (min=313, max=1086)
- 30 FPS, 7D 关节状态 + 双摄像头

### 镜像模式与起始位姿冲突

**问题**：用户按 H 让被控臂回起始位姿，但示教臂手握着在另一个位置 → CAN 镜像立即把被控臂拉回示教臂位置。

**原因**：硬件镜像模式下，只要两只臂位置不同，CAN 总线上的位置同步会立即生效，无法从软件层面阻止。

**解决**：移除 P/H 自动起点管理。实际操作流程改为：人工对齐示教臂和被控臂 → 空格开始录。

### 训练脚本：LeRobot v0.4.4 → v0.5.2 迁移

**CLI 参数变更**

| v0.4.4 | v0.5.2 |
|---|---|
| `python -m lerobot.scripts.train` | `python -m lerobot.scripts.lerobot_train` |
| `--lr=1e-4` | `--optimizer.lr=1e-4` |
| `--policy.feedforward_dim` | `--policy.dim_feedforward` |
| 无 | `--policy.repo_id=...` (必填) |
| 无 | `--policy.push_to_hub=false` (本地训练必加) |
| 无 | `--dataset.image_transforms.enable=true` |

**parser 模块变更**
- `parse_args` → `parse_arg`（单个参数解析）
- 配置类从分立的 `.py` 文件整合到 `TrainPipelineConfig` 统一 dataclass

### feature_utils.py Bug 修复

**问题**：训练启动时 `KeyError: 'names'` 崩溃。

**定位**：`lerobot/utils/feature_utils.py:153`
```python
names = ft["names"]  # LeRobotDataset v3.0 的 video 特征没有 "names" 字段
```

**原因**：v3.0 格式中 video 类型的 shape 已是 (C, H, W)，不需要 `names` 标注通道顺序。但代码假设所有 image/video 特征都有该字段（为了兼容 v2.0 的 HWC→CHW 转换）。

**修复**：
```python
names = ft.get("names")
if names and names[2] in ["channel", "channels"]:
    shape = (shape[2], shape[0], shape[1])
```

**注意**：这是在 lerobot 源码上的 patch（editable install），换机器或重装后需重新打。

### AV1 视频编码与 torchcodec 不可用

**现象**：训练日志中 `torchcodec is not available in your platform, falling back to pyav`。

**影响**：视频解码走 CPU（pyav），可能成为训练瓶颈。

**状态**：当前可接受，暂时不需要 GPU 加速解码。如果后续数据集变大（>100 episodes），可考虑预处理为图像序列。

### torchvision 视频解码废弃警告

**警告**：
```
UserWarning: The video decoding and encoding capabilities of torchvision are deprecated from version 0.22
and will be removed in version 0.24. We recommend that you migrate to TorchCodec
```

**状态**：这是 torchvision 的内部视频解码器（非核心路径），不影响功能。实际的视频解码走的是 pyav。

---

## 2026-05-09 — 训练完成与过拟合分析

### 训练结果

100,000 步训练完成，耗时 9h 42min（RTX 3060）。

| Checkpoint | Train Loss | 验证 MSE (5 集) | 结论 |
|---|---|---|---|
| 20,000 步 | 0.099 | **0.015** | ✅ 最佳模型 |
| 40,000 步 | — | — | 待评估 |
| 60,000 步 | — | — | 待评估 |
| 80,000 步 | — | — | 待评估 |
| 100,000 步 | 0.037 | 0.145 | ❌ 严重过拟合 |

### 过拟合分析

**现象**：训练 loss 从 0.099 → 0.037（持续下降），但验证 MSE 从 0.015 → 0.145（反而涨了 10 倍）。

**根本原因**：数据量不足（仅 20 episodes，9,506 frames）+ 模型容量过大（54M 参数）+ 训练步数过多（100K 步）。

模型从 20K 步之后开始记忆训练数据的具体轨迹，失去了泛化到未见过的 episode 的能力。典型的小数据过拟合特征：train loss 下降但 val error 上升。

**关键数据**：

```
20K checkpoint（最佳）各关节 MSE：
  j1:      0.0044   ✅
  j2:      0.0273   ⚠️  主力关节，最大误差
  j3:      0.0302   ⚠️  主力关节
  j4:      0.0178
  j5:      0.0180
  j6:      0.0087
  gripper: 0.00007  ✅  接近完美

100K checkpoint（过拟合）各关节 MSE：
  j1:      0.0041   ✅  这个反而更好（j1 动作简单）
  j2:      0.4517   ❌  严重退化
  j3:      0.2553   ❌  严重退化
  j4:      0.1081   ❌
  j5:      0.1106   ❌
  j6:      0.0840   ❌
  gripper: 0.0008   ⚠️  稍差但可接受
```

过拟合最严重的是 j2/j3（大臂主关节），从 0.03 飙升到 0.25-0.45。这两个关节运动幅度最大、轨迹最复杂，模型不是学会了泛化而是背下了训练轨迹。

### 改进方向

1. **补采数据**：从 20 集增加到 50-100 集，变化瓶子位置、角度、抓取方式
2. **早停**：20K 步左右停止训练，或者基于验证 loss 做 early stopping
3. **增大 dropout**：从 0.1 增到 0.2-0.3
4. **减小模型**：降低 dim_model（512→256）、减少层数（4→2）
5. **数据增广**：当前已启用，可增加增广强度（更大的 affine 范围、颜色抖动）

### 当前可用模型

- **推荐部署**：20K checkpoint
  ```
  outputs/train/piper_bottle_grasp/checkpoints/020000/pretrained_model
  ```
- MSE ~0.015 rad（约 0.12 rad ≈ 7° 平均关节误差）
- 部署时建议 `--velocity-pct 30` 低速测试，观察动作质量

### 推理脚本适配 v0.5.2

**API 变更**：

| v0.4.4 | v0.5.2 |
|---|---|
| `policy.predict_action(obs, chunk_size)` | `policy.select_action(norm_batch)` 或 `policy.predict_action_chunk(norm_batch)` |
| 手动归一化 | `NormalizerProcessorStep` + `UnnormalizerProcessorStep` pipeline |
| 单摄像头 | 双摄像头（wrist + global） |

**归一化 pipeline**：
- 使用 `make_pre_post_processors()` 或手动构建 `NormalizerProcessorStep`
- 统计量从 `dataset.meta.stats` 读取（训练时自动计算）
- 推理时：观测归一化 → 模型推理 → 动作反归一化 → 机械臂执行

**`select_action` vs `predict_action_chunk`**：
- `select_action`：管理内部队列，每次返回一个动作，适合连续实时控制
- `predict_action_chunk`：一次返回完整动作块，适合离线评估或批量执行
- 部署时按 SPACE 触发：先 `policy.reset()` 清空队列，再循环调用 `select_action` 执行完整轨迹

### 遗留问题

1. **训练数据量**：20 episodes 对 ACT 来说较少，建议后续补到 50-100 episodes
2. **模型多样性**：当前数据可能瓶子位置变化不够，推理时泛化能力待验证
3. **全局相机有效性**：USB 相机偶尔 V4L2 打开失败，后续可考虑直接使用 RealSense 替代或检查 USB 带宽
4. **AV1 编码速度**：采集时 AV1 编码 CPU 占用较高，如果影响实时性可考虑 H.264

---

## 2026-05-09（续）— 真机部署调试 + 数据重采 + 重新训练

### deploy.py Bug：真机不动

**问题**：修改 deploy.py 后真机部署，机械臂完全不动。dry-run 发现推理能跑通但输出的 delta 极小（~0.02 rad），policy 输出接近当前状态。

**根本原因（第一处 bug）**：`prepare_observation()` 有两个缺陷：
1. **没有加 batch 维度**（`.unsqueeze(0)`）—— LeRobot 的 preprocessor pipeline 内部有 `batch_to_transition` 转换，期望输入带 batch 维（`(1, 7)` 而非 `(7,)`）
2. **`device` 参数传了但完全没用到**—— tensor 留在 CPU 上，policy 在 GPU 上，`select_action` 时设备不匹配

**修复**：
```python
# 修复前
obs["observation.state"] = torch.from_numpy(np.asarray(state, dtype=np.float32))

# 修复后
obs["observation.state"] = torch.from_numpy(
    np.asarray(state, dtype=np.float32)
).unsqueeze(0).to(device)
```

两个图像 tensor 同理加了 `.unsqueeze(0).to(device)`。

**验证**：`--dry-run --debug-actions --debug-every 1` 确认 pipeline 恢复正常输出合理动作值。

### deploy.py Bug：夹爪限位过紧

**问题**：`PIPER_GRIPPER_MAX_M = 0.08` 但数据中夹爪最大开到 0.099 m，部署时会被截断导致夹爪无法完全张开。

**修复**：`PIPER_GRIPPER_MAX_M = 0.10`

### 训练数据删除与重采

**背景**：上一次的 20 集数据（9506 帧）被用户手动删除。

**重采**：重新录制 20 集，6,257 帧（均 313 帧/集）。

**数据质量验证**：
```
20 episodes, 6257 frames
j1: [-0.57, 0.59]     j4: [-1.26, 0.91]
j2: [ 0.00, 1.96]     j5: [-0.77, 1.17]
j3: [-2.26, 0.00]     j6: [-1.75, 1.20]
gripper: [0.00, 0.099]
```
- 起点一致性良好（所有 episode 起始关节角非常接近），手动回零操作稳定
- 终点轨迹有足够多样性（j2: 0.80-1.57, j3: -0.74 ~ -1.74），瓶子位置有变化
- 夹爪 49% 帧张开，抓取动作完整

### train.sh 问题修复

**问题 1**：`python` 命令在 nohup 环境下找不到。
**修复**：改为 `python3`。

**问题 2**：nohup 非交互 shell 使用系统 `/usr/bin/python3`，没有 lerobot。
**修复**：使用 conda 完整路径并加 `PYTHONPATH=` 前缀：
```bash
PYTHONPATH= ~/miniconda3/envs/piper_act/bin/python3 -m lerobot.scripts.lerobot_train
```

**问题 3**：上次训练输出目录 `outputs/train/piper_bottle_grasp/` 还存在，LeRobot 校验拒绝覆盖。
**修复**：`rm -rf` 删除旧目录后重建。

**问题 4**：`--steps` 从 100000 改为 20000，匹配 20K 早停策略。

### 训练重启动

- 步数：20,000（约 2 小时）
- 速度：~3.0-3.4 step/s（比上次稍快，数据量相近）
- 策略：吸取上次过拟合教训，20K 步直接出 checkpoint 停止

### 真机部署：机械臂原地抖动，无法抓取

**现象**：模型训练完成后真机部署，机械臂一抖一抖但不抬起来，只在原地小幅振荡。dry-run 显示 delta 仅 ~0.02 rad，模型输出的目标位置几乎等于当前位置。

**排查过程**：

1. 确认推理 pipeline 正常（`prepare_observation` 的 batch dim / device 已修复）
2. 用全零 dummy 输入测试 pipeline —— 模型输出 `j2→0.251`，证明有能力产
   生大幅动作
3. 用数据集 frame 0 测试 —— 模型输出误差 0.038 rad，在数据上表现正常
4. 检查预测的 20 步 chunk —— 发现所有 20 步动作几乎相同，没有运动轨迹
5. 检查组块真值 —— 发现数据集中每条的**前 20+ 帧完全相同**，机械臂没有移动

**根本原因：训练数据中的"死区"**

录音时按空格开始后，操作员会停顿 1.5-2 秒才开始抓取动作。数据集中每个 episode 开头有 50-70 帧（1.7-2.3s）状态完全不动，末尾也有 5-45 帧停顿。

ACT 模型的 chunk_size=20，第一次推理时看到画面+状态 → 预测接下来 20 步 → 数据中前 20 步全是死区 → 模型学会"这个场景下不要动"→ 输出平线。

更关键的是 ACT 的推理循环：`select_action()` 先将 20 步动作入队，逐步执行，20 步后再用最新观测重新预测。但如果 20 步全是死区（delta ≈ 0），机械臂没有实际移动，画面和状态都没有改变，下一次预测仍然看到相同的入 = 再一次输出死区 chunk。**模型永远被困在"开头不动"的状态里出不来**，根本无法推进到后续的靠近/抓取阶段。

这不是模型没学会抓取，而是起点到抓取之间被一个"死区门槛"卡死了。

**解决方案：剪掉死区帧重训**

写脚本逐 episode 检测首尾死区（state delta > 0.02 rad 判定为"有动作"），剪掉首尾死帧（开头保留 5 帧预备，结尾保留 2 帧收尾）：

```
Before: 6257 frames
After:  4971 frames (1286 removed, ~20%)

逐 episode 示例:
  Ep 0: 293→246 fr (dead_start=49, dead_end=5)
  Ep 1: 324→237 fr (dead_start=60, dead_end=34)
  Ep 3: 420→336 fr (dead_start=70, dead_end=21)
```

剪完后前 20 帧就有实质动作（j4/j5 从第 4 帧开始动，j2/j3 从第 16 帧开始动），模型第一次预测的 chunk 就会包含"靠近瓶子"，启动运动后新观测会推进到抓取阶段。

**教训**：
- 采集时按空格后应立即开始动作，不要停顿
- 采集结束时动作完立即按空格停止，不要多余停顿
- 或者：用脚本自动剪掉死区再训练
- 后续可在采集程序中加入"检测到动作自动开始录"的逻辑，避免人为死区

### VAE 后验坍缩：模型对不同输入输出同一个动作

**现象**：死区修复后 dry-run 显示 delta 从 0.02 → 1.5 rad，模型确实能预测大动作了。但机械臂执行时卡在半空——因为模型不管看到什么画面/状态，输出的 20 步 chunk 都是**完全相同的平线**：

```
Frame 0  (起点):   Pred: j2=1.484  j3=-0.951  j5=0.128  ← 同一个
Frame 50 (途中):   Pred: j2=1.484  j3=-0.951  j5=0.128  ← 同一个
Chunk range:       j2=0.0000  ← 20步完全一样
```

**根本原因**：ACT 使用 VAE 结构（`use_vae=true`），训练损失 = L2 重建损失 + `kl_weight * KL散度`。`kl_weight=10.0` 在高容量模型（54M）+ 小数据（4971 帧）下，KL 散度项压倒重建损失。VAE 的 latent 被强制压到先验分布（零均值高斯），decoder 只能输出一个与输入无关的"平均轨迹"。

这是经典的 **posterior collapse（后验坍缩）**——latent code 不再编码任何观测信息，模型退化为一个固定的映射：不管看到什么，输出都一样。

**修复**：`kl_weight` 从 10.0 降到 1.0，减弱 KL 正则化强度，让 latent 保留观测信息。这是 VAE + 小数据的常见调参策略。

**教训**：
- VAE 的 kl_weight 是敏感超参，数据越少越要调低
- 如果 kl_weight 太高，模型宁可让 latent 坍缩也不优化重建——因为"预测均值"已经能在 KL=0 时拿到不错的 loss
- 判断标准：对不同输入帧（frame 0 vs frame 50）预测的 chunk 应该不同，如果完全相同就是坍缩了

### 当前遗留风险

1. **数据量仍然偏少**：20 集 / 4971 帧对 ACT（54M 参数）来说偏小，即使降低 kl_weight，也可能出现其他形式的过拟合
2. **chunk 可能仍然是平线或变化很小**：ACT 的 VAE decoder 输出 chunk 的帧间变化天然较小（动作序列是连续的），需要在实际部署中观察 chunk 内的动作变化幅度
3. **相机视角一致性**：部署时相机位置必须和采集时一致，否则视觉特征完全不对
4. **起点位置依赖**：模型只在"零位起点附近"的数据上训练过，如果部署时起点偏差太大可能出问题
5. **AV1 编码/解码**：当前使用的 AV1（libsvtav1）是软件编解码，CPU 占用较高但 GPU 不支持硬件加速，大数量训练时可能成为瓶颈

---

## 2026-05-09（续 2）— kl_weight=1.0 仍坍缩 + 关闭 VAE 重训

### kl_weight=1.0 仍然 VAE 后验坍缩

**现象**：kl_weight 从 10.0 降到 1.0 后训练 20K 步，dry-run 显示 max_arm_delta=1.485 rad（有大动作），但：

1. **Chunk 内 20 步完全平线**：每步的 j1-j6 + gripper 范围都是 0.0000，20 步是同一个值
2. **不同输入输出完全相同**：用 frame 0、frame 200、frame 3000（来自不同 episode，关节状态完全不同）分别预测，输出的 chunk 完全一样（mean_abs_diff=0.0000）

这三帧的状态差异巨大：
```
Frame 0:    j2=-0.002  j3=-0.003  (episode 起点)
Frame 200:  j2=1.631   j3=-1.101  (抓取中途)
Frame 3000: j2=0.991   j3=-1.447  (另一条 episode)
```
模型不管看到什么都输出 `j2=1.484, j3=-0.951`，一个固定的"平均轨迹"。

**根本原因**：数据量（4971 帧 / 20 episodes）对 VAE（54M 参数）来说实在太少。即使 kl_weight=1.0，KL 正则化仍然压倒重建损失。这不是调参能解决的——在小数据下，VAE 学到最优策略就是忽略 latent code 直接输出均值。

### 修复：关闭 VAE

`--policy.use_vae=false`，将 ACT 变成**确定性 transformer**：
- 参数量：54M → 41M（去掉 VAE 的 encoder/decoder 结构）
- 不再有 KL 散度损失，只优化 L2 重建
- 适合小数据场景，不会发生后验坍缩
- 缺点是失去了多模态动作分布的建模能力，但对 20 集的单一抓取任务影响不大

**当前状态**：`use_vae=false` 训练已启动（PID 2526234），预计 ~55 分钟完成 20K 步。

旧模型已重命名为 `outputs/train/piper_bottle_grasp_v2_kl1.0` 保留对比。

### 教训

- VAE + 小数据 = 极高风险的后验坍缩，调低 kl_weight 可能不够
- 判断坍缩不能只看 delta 绝对值（1.485 rad 看起来很大），要看：
  1. **Chunk 内部是否有变化**（range > 0）
  2. **不同输入是否产生不同输出**（mean_abs_diff > 0）
- 如果两个都是 0，说明 VAE 已经坍缩，模型退化成一个固定映射
- 小数据下优先用确定性模型（`use_vae=false`），数据够多（>100 episodes）再考虑 VAE

---

## 2026-05-11 — 确定性模型仍坍缩 + 数据集不一致诊断 + 重建修复

### 确定性模型（use_vae=false）仍然输出恒定值

**现象**：`use_vae=false`, lr=1e-4 训练 20K 步后（loss=0.222, grad_norm=2.181），诊断发现：

1. **不同输入输出完全相同**：用 4 种完全不同的输入（frame 0、frame 200、全零图像+全零状态、全一图像+全一状态），模型输出完全一致（mean_abs_diff=0.000000）。

2. **Chunk 内 20 步全部平线**：无论哪个输入，20 个时间步的动作完全一样（per-pos variation=0.000000）。

3. **Encoder 输出差异极小**：即使全零 vs 全一这种极端输入，encoder 输出差异仅 ~5e-6。

4. **随机初始化模型同样不敏感**：新鲜 Xavier 初始化的模型，不同输入输出差异也仅 0.001。

**根本原因**：20 episodes / 4971 frames 对 41M 参数、602 token 的 transformer 来说数据量不足。loss 在 200 步（0.269）后就基本停滞，在 0.20-0.27 范围内振荡，模型只是在输出一个"平均最优轨迹"而不是学习输入→输出的映射。

### 数据集裁剪后 metadata 不一致

**问题**：之前用 `scripts/rebuild_trimmed_dataset.py` 的简单版本裁剪死区时，只修改了 parquet 数据行和 info.json 的 total_frames，没有完整重建 episode metadata。导致：

- **20/20 episode** 的 `frame_index` 不从 0 开始（如 Ep 0 从 44 开始）
- **20/20 episode** 的 `timestamp` 不从 0 开始（如 Ep 0 从 1.467s 开始）
- **20/20 episode** 的 `dataset_from_index` / `dataset_to_index` 与实际 global index 偏移

这会导致 LeRobot 的 episode 边界检查、视频 timestamp 解码、evaluation 等出现不一致。虽然 action chunk 组装用的是位置索引（不受影响），但这是一个隐藏的数据结构炸弹。

**诊断命令**：
```python
import numpy as np, pyarrow.parquet as pq
data = pq.read_table('data/lerobot_dataset/data/chunk-000/file-000.parquet')
for ep in sorted(set(data.column('episode_index').to_pylist())):
    mask = data.column('episode_index').to_pylist() == ep
    frames = np.array(data.column('frame_index').to_pylist())[mask]
    ts = np.array(data.column('timestamp').to_pylist())[mask]
    print(f'Ep {ep}: frame_index_start={frames[0]}, timestamp_start={ts[0]:.3f}s')
```

### 修复：完整重建数据集

用 LeRobot 官方 API（`create()` + `add_frame()` + `save_episode()` + `finalize()`）逐帧从源数据集读取并写入新数据集，自动生成正确的 metadata、frame_index、timestamp、video chunks。

```bash
python3 scripts/rebuild_trimmed_dataset.py \
  --input-root data/lerobot_dataset \
  --output-root data/lerobot_dataset_rebuilt \
  --motion-threshold 0.005 \
  --preroll-frames 5 \
  --tail-frames 8
```

结果：
- 4968 帧（比原 4971 少 3 帧，因为重建过程中重新计算 trim 边界）
- **0/20 episode** 有问题（全部 frame_index/timestamp/metadata 正确）

### 训练策略调整

1. **chunk_size 降为 10**：`n_action_steps=20` 意味着每 20 步才重新预测一次，开环周期太长（~0.67s at 30fps）。降到 10 后更短的开环窗口，模型更容易适应。

2. **学习率提高**：`optimizer_lr` 从 1e-4 提高到 5e-4，`optimizer_lr_backbone` 从 1e-5 提高到 1e-4（10 倍），解决 backbone 几乎不更新的问题。

3. **训练步数增加**：从 20K 到 50K，每 10K 保存 checkpoint。

4. **部署时 replan-every-step**：绕过 ACT 的 action queue，每一步都重新预测完整 chunk 并只取第一步，避免开环积累误差。

### 新脚本总览

| 脚本 | 功能 |
|---|---|
| `scripts/rebuild_trimmed_dataset.py` | 重建裁剪后的 LeRobot dataset（完整 API） |
| `scripts/analyze_dataset_motion.py` | 诊断数据集运动/静止段/跳跃/对齐 |
| `scripts/eval_all_checkpoints.py` | 批量评估所有 checkpoint 的 MSE |
| `scripts/plot_policy_rollout_on_dataset.py` | 在数据集上绘制 policy rollout 对比 |
| `training/train_act_chunk10.sh` | chunk=10, use_vae=false, lr=5e-4 |
| `training/train_act_chunk20.sh` | chunk=20, use_vae=false, lr=5e-4 |
| `training/train_act_chunk40.sh` | chunk=40, use_vae=false, lr=5e-4 |

### 当前遗留风险

1. **数据量核心瓶颈**：即使修好数据集、调优超参，20 episodes / ~5000 frames 对 ACT 来说仍然偏少。更高的 lr 和更多步数可能让模型学习到一些输入→输出映射，但不能保证泛化。理想情况需要 50-100 episodes。

2. **chunk=10 vs chunk=20 取捨**：chunk=10 开环短、更稳定，但预测视野更短；chunk=20 能规划更长轨迹但更容易漂移。需要用实际部署效果来判断。

3. **部署安全**：首次部署新模型时先用 `--dry-run` + `--debug-actions --debug-every 1` 验证，确认 target_diff_from_last_target 在变化后再真机运行。
