# Cube 方块 ACT 训练全记录

日期：2026-05-27 ~ 2026-05-29

## 概述

从基线到最终 64 条 demo 模型，共 7 轮训练，逐轮增加数据量和复杂度。

所有训练共享以下配置：
- 策略：ACT | chunk_size=10 | n_obs_steps=1 | vision_backbone=resnet18
- 归一化：MEAN_STD | 图像增强：关闭 | 设备：cuda
- 冻结代码：未修改 adapter/deploy/record/schema/gripper/camera/normalization/close-latch/TEST-D/full-e2e 逻辑
- 提交哈希：`6d97aea`

---

## Train 0 — Blue R0 D128 基线

| 项目 | 值 |
|------|-----|
| 数据集 | `train_blue_r0_global16`（16 条，2073 帧） |
| 数据组成 | 蓝色方块，R0 底板，4 位置×4 |
| dim_model | 128 |
| 步数 | 5000 |
| 最终 Loss | 0.142 |
| 目的 | 最小基线，验证训练管线可用 |

输出目录：`outputs/train/act_cube_blue_r0_d128/`

---

## Train 1 — Blue R0 D256

| 项目 | 值 |
|------|-----|
| 数据集 | `train_blue_r0_global16`（16 条，2073 帧） |
| 数据组成 | 蓝色方块，R0 底板，4 位置×4 |
| dim_model | 256 |
| 步数 | 5000 |
| 最终 Loss | 0.172 |
| 目的 | dim128→dim256，验证容量提升后仍能正常收敛 |

输出目录：`outputs/train/act_cube_blue_r0_d256/`

---

## Train 2 — Blue+Purple R0 D256

| 项目 | 值 |
|------|-----|
| 数据集 | `train_blue_purple_r0_global32`（32 条，4068 帧） |
| 数据组成 | blue_r0 16条 + purple_r0_balanced 16条 |
| dim_model | 256 |
| 步数 | 8000 |
| 最终 Loss | 0.160 |
| 目的 | 增加颜色泛化（蓝→蓝+紫），验证多颜色数据可收敛 |

输出目录：`outputs/train/act_cube_blue_purple_r0_d256/`

---

## Train 3 — Blue R0+R90 D256

| 项目 | 值 |
|------|-----|
| 数据集 | `train_blue_r0_r90_global32`（32 条，4055 帧） |
| 数据组成 | blue_r0 16条 + blue_r90 16条 |
| dim_model | 256 |
| 步数 | 8000 |
| 最终 Loss | 0.163 |
| 目的 | 增加底板方向泛化（R0→R0+R90），验证多底板数据可收敛 |

输出目录：`outputs/train/act_cube_blue_r0_r90_d256/`

---

## 双摄实验 — Blue R0 Dual D128（clean16）

| 项目 | 值 |
|------|-----|
| 数据集 | `blue_block_r0_dual_clean16`（16 条，2073 帧） |
| 数据组成 | 蓝色方块，R0 底板，4 位置×4 |
| 相机 | global_rgb + wrist_rgb |
| dim_model | 128 |
| 步数 | 5000 |
| 最终 Loss | 0.153 |
| 目的 | 验证双摄管线，使用原始 clean16 数据集 |

输出目录：`outputs/train/act_cube_blue_r0_dual_d128_run1/`

---

## 双摄实验 — Blue R0 Dual D128（current16）

| 项目 | 值 |
|------|-----|
| 数据集 | `blue_block_r0_dual_current16`（16 条，2545 帧） |
| 数据组成 | 蓝色方块，R0 底板，4 位置×4，当前场景 |
| 相机 | global_rgb + wrist_rgb |
| dim_model | 128 |
| 步数 | 5000 |
| 最终 Loss | 0.143 |
| 离线 MSE | 0.001346（4 个 episode，cuda） |
| 目的 | 验证当前场景双摄管线，为 64 条双摄训练做准备 |

输出目录：`outputs/train/act_cube_blue_r0_dual_current_d128_run1/`

---

## Train 4（最终）— Pos4 Color R0+R90 D256 平衡 64 条

| 项目 | 值 |
|------|-----|
| 数据集 | `cube_64_global`（64 条，8045 帧） |
| 数据组成 | blue_r0 16 + purple_r0 16 + blue_r90 16 + purple_r90 16 |
| 位置平衡 | P1=P2=P3=P4=16 |
| 相机 | global_rgb（单摄） |
| dim_model | 256 |
| 步数 | 15000 |
| 保存频率 | 1000 |
| 最终 Loss | 0.132 |
| 离线 MSE | 0.000733（8 个 episode，cpu） |
| 最大关节误差 | j2=0.003075 |

输出目录：`outputs/train/act_cube_pos4_color_r0_r90_d256/`

离线评估详情：
- 各关节 MSE：j1=0.000770, j2=0.003075, j3=0.000686, j4=0.000033, j5=0.000413, j6=0.000145, gripper=0.000011
- 代表性 rollout（ep 0 blue_r0, ep 16 purple_r0, ep 32 blue_r90, ep 48 purple_r90）均正常

---

## 进近阶段模型（补充）

| 项目 | 值 |
|------|-----|
| 数据集 | `cube_64_global_approach`（64 条） |
| 说明 | Train 4 数据截取进近阶段（裁掉静止前缀和夹爪闭合后的帧） |
| 相机 | global_rgb（单摄） |
| dim_model | 256 |
| 步数 | 8000 |
| 目的 | 仅进近阶段，用于快速验证进近轨迹方向 |

输出目录：`outputs/train/act_cube_approach64_global_current_d256_run1/`

---

## 损失趋势

| 训练轮次 | Demos | Dim | 步数 | Final Loss |
|---------|-------|-----|------|------------|
| Train 0 (blue R0) | 16 | 128 | 5K | 0.142 |
| Train 1 (blue R0) | 16 | 256 | 5K | 0.172 |
| Train 2 (+purple) | 32 | 256 | 8K | 0.160 |
| Train 3 (+R90) | 32 | 256 | 8K | 0.163 |
| Dual clean16 | 16 | 128 | 5K | 0.153 |
| Dual current16 | 16 | 128 | 5K | 0.143 |
| **Train 4 (最终)** | **64** | **256** | **15K** | **0.132** |

损失不直接可比（数据集大小/分布不同），但趋势说明：数据量增加、步数增加后损失持续下降。

---

## 已知问题

- **Train 4 真机表现不佳**：离线 MSE 仅 0.000733，但真机部署时机械臂定位不准（"在空中不知道该抓哪里"）。可能原因：OOD 起始位姿、相机视角偏差、sim-to-real gap。
- **双摄 approach 模型（act_cube_approach64_dual_current_d256）训练失败**：J2 仅到 0.67，基本不动。原因未定位，建议重新训练。
