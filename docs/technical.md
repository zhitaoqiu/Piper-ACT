# Piper ACT Bottle Grasp — 技术文档

## 目录

1. [系统架构](#1-系统架构)
2. [算法原理](#2-算法原理)
3. [数据流与格式](#3-数据流与格式)
4. [操作 SOP](#4-操作-sop)
5. [踩坑记录](#5-踩坑记录)

---

## 1. 系统架构

### 硬件拓扑

```
┌──────────────────────────────────────────────────┐
│                    工控机                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │ CAN (can0)│  │ USB 3.0  │  │ USB 2.0         │ │
│  └────┬─────┘  └────┬─────┘  └───────┬──────────┘ │
│       │             │                │             │
└───────┼─────────────┼────────────────┼─────────────┘
        │             │                │
   ┌────┴────┐   ┌────┴────┐     ┌────┴────┐
   │ 示教臂   │   │ D435i   │     │ SN0002  │
   │ (Leader) │   │ 腕部相机 │     │ 全局相机 │
   └────┬────┘   └─────────┘     └─────────┘
        │
   CAN 总线共享
        │
   ┌────┴────┐
   │ 被控臂   │
   │(Follower)│
   └─────────┘
```

### 镜像模式原理

Piper 机械臂支持**硬件级镜像模式**：示教臂和被控臂连接在同一条 CAN 总线上，示教臂通过 CAN 广播自身关节位置，被控臂在硬件固件层面自动跟随，无需上位机转发。

- **优点**：零延迟跟随，无需软件处理
- **约束**：数据采集时需注意被控臂的状态就是示教臂的相同状态，因此只需读取被控臂的 `get_joint_positions()`
- **起始位姿问题**：如果上一轮采集结束位置与下一轮起始位置不同，示教臂和被控臂会有位置偏差，因为被控臂跟随的是当前示教臂位置。解决方案：人工手动把示教臂拖回起始位置对齐即可

### 软件架构

```
┌─────────────────────────────────────────────────────────┐
│                     应用层                               │
│  teleop/data_collector.py    inference/deploy.py        │
│  采集示教数据                   推理部署                  │
├─────────────────────────────────────────────────────────┤
│                     算法层                               │
│  LeRobot ACT Policy            Normalization Pipeline    │
│  (ResNet18 + Transformer)      (Mean/Std 归一化)        │
├─────────────────────────────────────────────────────────┤
│                     驱动层                               │
│  hardware/piper_wrapper.py     camera/rs_camera.py      │
│  (Piper SDK 封装)              (RealSense + USB)        │
├─────────────────────────────────────────────────────────┤
│                     硬件抽象层                            │
│  piper_sdk_py_driver/          pyrealsense2 / OpenCV    │
│  (CAN 通信)                    (相机采集)                │
└─────────────────────────────────────────────────────────┘
```

### 模块职责

| 模块 | 文件 | 职责 |
|---|---|---|
| 机械臂驱动 | `hardware/piper_wrapper.py` | CAN 通信、使能/失能、关节读写、安全限位 |
| 相机驱动 | `camera/rs_camera.py` | RealSense + USB 相机初始化、帧读取、自动设备扫描 |
| 数据采集 | `teleop/data_collector.py` | 键盘控制、数据记录为 LeRobot v3.0 格式 |
| 训练 | `training/train.sh` | ACT 模型训练、checkpoint 管理 |
| 推理部署 | `inference/deploy.py` | 加载模型 + 归一化 pipeline、实时控制机械臂 |
| 离线评估 | `inference/eval.py` | 在数据集上评估模型精度 |

---

## 2. 算法原理

### ACT (Action Chunking with Transformers)

ACT 是一种基于 Transformer 的模仿学习算法，核心思想是**预测一个连续的动作块**而非单个动作，以减少累积误差。

#### 网络结构

```
输入                         输出
┌─────────────────┐         ┌──────────────┐
│ observation.state│         │              │
│ (7D 关节状态)     │         │   action     │
├─────────────────┤         │ (chunk_size, │
│ wrist_rgb       │         │  7D 动作)    │
│ (3,480,640)     │         │              │
├─────────────────┤  ACT    │  每个动作 =   │
│ global_rgb      │ Policy  │  j1..j6 +    │
│ (3,480,640)     │         │  gripper     │
└─────────────────┘         └──────────────┘
```

**编码器（Encoder）**：
1. **视觉编码**：ResNet18 (ImageNet 预训练) 将每张 480×640 RGB 图编码为特征向量
2. **状态编码**：7D 关节状态通过线性层映射到 d_model 维度
3. **Transformer Encoder** (n_encoder_layers=4)：融合视觉特征 + 状态特征 + 位置编码

**解码器（Decoder）**：
1. 可学习的 query embeddings（类似 DETR 的目标查询）
2. **Transformer Decoder** (n_decoder_layers=4)：交叉注意力到编码器输出
3. 输出头：线性层映射到 7D 动作空间

**VAE 组件**（use_vae=true）：
- 额外的 VAE Encoder（Transformer）用于对完整动作序列的潜在表示建模
- 训练时：VAE Encoder 编码真实动作序列 → 潜在变量 z，Decoder 基于 z 和输入预测动作
- 推理时：z 设为全零（先验均值），Decoder 基于输入预测动作
- Loss = reconstruction_loss + kl_weight × KL(后验 || 先验)

#### 推理时 Chunk 管理

```
时间 →   t0   t1   t2  ...  t20  t21  t22
         │                  │
         └─ 模型推理 ──────┘
            预测 chunk_size=20 个动作
            ├─ 动作 0:  执行 ✓
            ├─ 动作 1:  执行 ✓
            ├─ ...
            └─ 动作 19: 执行 ✓
                             │
                             └─ 再次推理，预测下一段
```

- `chunk_size=20`：模型一次预测 20 步动作
- `n_action_steps=20`：执行完 20 步后重新推理（本项目设置等同于 chunk_size，即用完再推理）
- 推理使用 `policy.select_action()` 管理内部队列，或直接调用 `predict_action_chunk()` 获取完整块

#### 归一化策略

所有数据使用 **MEAN_STD 归一化**（与原始 ACT 一致）：

| 特征类型 | 归一化方式 | 说明 |
|---|---|---|
| STATE (关节状态) | (x - mean) / std | 每个关节维度独立统计 |
| VISUAL (图像) | (x - mean) / std | ImageNet 统计量 (mean, std) |
| ACTION (动作) | (x - mean) / std | 每个关节维度独立统计 |

归一化统计量在数据集创建时计算并保存在 `data/lerobot_dataset/meta/stats.json`。推理时使用相同的统计量。

---

## 3. 数据流与格式

### 采集数据流

```
1. 用户按空格
      ↓
2. 采集线程启动（control_rate=30Hz）
      ↓
   ┌─ 读被控臂关节状态 (7D) → observation.state
   ├─ 读腕部相机 (640×480 BGR) → observation.images.wrist_rgb
   └─ 读全局相机 (640×480 BGR) → observation.images.global_rgb
      ↓
3. action[t] ← state[t+1]（下一帧的关节状态作为当前帧的 action）
      ↓
4. 写入 LeRobot 数据集
   ├─ observation.state → Parquet 文件
   ├─ action → Parquet 文件
   ├─ observation.images.*.rgb → MP4 视频文件 (AV1 编码)
   └─ meta/ → 元数据
      ↓
5. 用户按空格 → 保存 episode
```

### LeRobot v3.0 数据集结构

```
data/lerobot_dataset/
├── meta/
│   ├── info.json        # 数据集元信息（特征定义、FPS）
│   ├── stats.json       # 归一化统计量（mean、std、min、max 等）
│   ├── episodes/        # episode 元数据
│   └── tasks.parquet    # 任务描述
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet   # 数值数据（state, action, timestamp）
│       ├── ...
│       └── episode_000019.parquet
└── videos/
    └── chunk-000/
        ├── observation.images.wrist_rgb/
        │   ├── episode_000000.mp4   # 腕部视频 (AV1, 30fps)
        │   └── ...
        └── observation.images.global_rgb/
            ├── episode_000000.mp4   # 全局视频 (AV1, 30fps)
            └── ...
```

### 特征定义

```json
{
    "observation.state": {
        "dtype": "float32",
        "shape": [7],
        "names": ["j1", "j2", "j3", "j4", "j5", "j6", "gripper"]
    },
    "action": {
        "dtype": "float32",
        "shape": [7],
        "names": ["j1", "j2", "j3", "j4", "j5", "j6", "gripper"]
    },
    "observation.images.wrist_rgb": {
        "dtype": "video",
        "shape": [3, 480, 640]
    },
    "observation.images.global_rgb": {
        "dtype": "video",
        "shape": [3, 480, 640]
    }
}
```

关节角单位 rad，值域通过安全限位保持在 [-3.14, 3.14] 内。

### 推理数据流

```
1. 按空格
      ↓
2. 采集当前观测
   ├─ wrist_cam.read() → (H, W, 3) uint8 BGR
   ├─ global_cam.read() → (H, W, 3) uint8 BGR
   └─ robot.get_joint_positions() → [float] × 7
      ↓
3. 预处理
   ├─ image: uint8 → float32/255.0 → (C, H, W) → 加 batch → 移到 GPU
   └─ state: (7,) → (1, 7) tensor → 移到 GPU
      ↓
4. Preprocessor (归一化)
   ├─ 图像: (x - imagenet_mean) / imagenet_std
   └─ 状态: (x - state_mean) / state_std
      ↓
5. ACT Policy (GPU)
   ├─ ResNet18 编码图像
   ├─ Transformer 融合多模态
   └─ 输出: (1, 7) 归一化动作
      ↓
6. Postprocessor (反归一化)
   action = action * std + mean
      ↓
7. 安全裁剪
   joint[:6] ∈ [-3.14, 3.14]
   gripper ∈ [0.0, 0.035]
      ↓
8. robot.set_joint_positions() → 机械臂运动
      ↓
9. 重复直到 n_action_steps 全部执行
```

---

## 4. 操作 SOP

### 日常操作流程

#### 开机

```bash
# 1. 确认机械臂上电
# 2. 配置 CAN
sudo bash scripts/setup_can.sh
# 3. 验证硬件
conda activate piper_act
python3 test_hardware.py
```

#### 采集数据

```bash
conda activate piper_act
python3 teleop/data_collector.py
```

1. 按 **E** 使能
2. 手动把示教臂和被控臂回到固定起点
3. 把瓶子放到本条 episode 的初始位置
4. 按**空格**开始录制
5. 执行一次完整抓取
6. 按**空格**停止并保存
7. 保存后再手动回到固定起点，重复步骤 2-6，目标 50-100 条

**关键**：不同 bottle 位置、不同角度、不同抓取策略都要覆盖

#### 训练

```bash
conda activate piper_act
bash training/train.sh
# 后台执行:
nohup bash training/train.sh > /tmp/train_piper_act.log 2>&1 &
```

监控：
```bash
tail -f /tmp/train_piper_act.log
```

#### 评估

```bash
conda activate piper_act
python3 inference/eval.py \
    --checkpt outputs/train/piper_bottle_grasp/checkpoints/last/pretrained_model
```

评估报告示例解读：
```
Mean MSE across 3 episodes: 0.001234
Per-joint MSE:
  j1: 0.000856    <- 大关节误差通常较大
  j2: 0.001234
  j3: 0.000923
  j4: 0.000567    <- 小关节误差通常较小
  j5: 0.000432
  j6: 0.000398
  gripper: 0.000012  <- 夹爪开合比较确定
```

#### 部署

```bash
conda activate piper_act
python3 inference/deploy.py \
    --checkpt outputs/train/piper_bottle_grasp/checkpoints/last/pretrained_model
```

### 故障处理速查

| 症状 | 检查项 | 解决 |
|---|---|---|
| 机械臂不动 | `ip link show can0` | `sudo bash scripts/setup_can.sh` |
| cv2 报错 _ARRAY_API | NumPy 版本 | `pip install "numpy<2" --force-reinstall opencv-python` |
| 全局相机黑屏 | 设备探测 | `python3 teleop/data_collector.py --list-cameras`，然后用 `--global-camera N` 指定 |
| RealSense 无深度 | D435/D435i 兼容性 | 会自动回退到 RGB-only |
| 训练崩溃 KeyError:'names' | lerobot patch | 确认 `feature_utils.py` 的 patch 已打 |
| PYTHONPATH 污染 | ROS2 环境 | `unset PYTHONPATH` 后重试 |

---

## 5. 踩坑记录

### 5.1 双机械臂架构的演进

**最初设想**：示教臂和被控臂分别接 can0/can1，上位机读示教臂状态并转发指令给被控臂（软件转发模式）。

**实际发现**：用户插电后发现两只臂已经在同一 CAN 总线上，被控臂自动镜像示教臂动作。这是 Piper 的硬件级镜像模式，完全不需要软件转发。

**启发**：先确认硬件能力再设计软件方案，避免不必要的工作。

### 5.2 LeRobot 版本选择

- **v0.4.4**：最初选型，API 较简单但已过时
- **v0.5.2**：最终使用版本，API 更成熟但 CLI 和配置系统有较大变化
- 升级要点：`train` → `lerobot_train`、`parse_args` → `parse_arg`、归一化引入 processor pipeline

### 5.3 归一化与 feature_utils.py Bug

LeRobot v0.5.2 的 `dataset_to_policy_features()` 假设所有图像/视频特征都有 `names` 字段来标注通道顺序。但 v3.0 格式的 video 特征 shape 已经是 (C, H, W) 格式，不需要 `names` 字段。

**修复**：`ft["names"]` → `ft.get("names")`，并加空值检查。

这是 LeRobot 的一个已知兼容性 bug，可能在后续版本修复。

### 5.4 PYTHONPATH 与 ROS2 冲突

ROS2 安装后会在 `~/.bashrc` 中设置 PYTHONPATH，指向系统级 Python 包路径（如 `/opt/ros/humble/lib/python3.10/site-packages`）。即使激活 conda 环境，PYTHONPATH 优先级高于 conda 环境，导致 import 时加载系统旧版包。

**现象**：`import lerobot` 导入的是 `~/.local/lib/python3.10/site-packages/` 的旧版而非 conda 环境的最新版。

**解决方案**：在 conda 环境的 `activate.d/` 和 `deactivate.d/` 中添加 hooks。

### 5.5 CAN 初始化的教训

`sudo ip link set can0 up type can bitrate 1000000` 只需要在系统启动后执行一次（或者 CAN 线重新插拔后）。不需要每次运行程序都重新配置。

**注意**：如果 `setup_can.sh` 先 `down` 再 `up`，而机械臂正在运行中，会导致失能。生产环境中应避免不必要的 CAN 重启。

### 5.6 AV1 编码与视频性能

LeRobot v3.0 默认使用 AV1 编码存储视频。AV1 压缩率高但解码计算量大，在训练时 CPU 视频解码可能成为瓶颈。`torchcodec` 支持 GPU 加速解码但当前 Linux 环境下不可用（用 `pyav` 回退）。

如果训练速度明显受限于视频解码，可考虑：
- 减小图像分辨率
- 使用预处理脚本将视频转为图像序列

### 5.7 起始位姿管理

镜像模式下，软件只控制被控臂，示教臂不会被程序自动拖回起点。如果只让被控臂回到某个保存位姿，而示教臂还在另一个位置，CAN 镜像会立即覆盖被控臂位置：

```
follower 移动到固定起点 → CAN 镜像检测到 leader 位置不同 → follower 立刻跟随 leader
```

**结论**：当前流程不再提供 P/H 自动起点管理。每条数据开始前，人工把示教臂和被控臂一起摆到固定起点；部署真机抓取前也用同一个固定起点。

### 5.8 图像增广对训练速度的影响

启用 `dataset.image_transforms.enable=true` 后，每个 batch 需要 CPU 执行 ColorJitter / SharpnessJitter / RandomAffine 等操作。这会导致训练速度下降约 10-20%。如果 GPU 利用率未满，可以降低 `num_workers` 减少 CPU 竞争；如果 CPU 是瓶颈，考虑关掉增广。

---

## 附录

### A. 依赖版本锁定

```
python=3.10
lerobot==0.5.2  (editable install from ~/third_party/lerobot)
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
pyrealsense2>=2.55.0
numpy>=1.24.0,<2.0
datasets>=3.0.0
pyarrow>=14.0.0
av>=11.0.0
```

### B. 关键路径速查

| 路径 | 说明 |
|---|---|
| `~/miniconda3/envs/piper_act/` | Conda 环境 |
| `~/third_party/lerobot/` | LeRobot 源码（editable 安装） |
| `data/lerobot_dataset/` | 采集的数据集 |
| `outputs/train/piper_bottle_grasp/` | 训练输出 |
| `/tmp/train_piper_act.log` | 训练日志 |
