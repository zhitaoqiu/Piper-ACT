# P0: Full-Trajectory ACT Overfit 实验报告

**日期**: 2026-05-19  
**目标**: 固定位置 1 条 full trajectory（approach→descend→close→lift→place→release），Tiny ACT 过拟合验证

## 实验结论：PASS

ACT 成功学到了 7D 绝对位置动作（J1-J6 + gripper），包括夹爪闭合/释放时序。
chunk_size=50 是关键 — chunk_size=1 会导致 mean-action 坍缩。

---

## Step 1: 数据采集

| 项目 | 值 |
|------|-----|
| 数据集 | `data/lerobot_dataset_full_fixed_1ep/` |
| Episode 数 | 2 |
| 总帧数 | 727 |
| 图像 | wrist_rgb + global_rgb @ 30fps |
| 动作空间 | 7D [J1..J6, gripper] |

## Step 2: 数据选优

Episode 0 胜出：
- 更少的 teleop 停顿（16 vs 17 clusters）
- J2 提瓶幅度更大（ΔJ2=0.216 vs 0.191）
- approach→close 过渡更紧凑

**关键修正**: 夹爪判断阈值从绝对阈值（0.02/0.06）改为动态阈值。瓶子有物理宽度，夹爪闭合到瓶身即停止（~0.046m），不会到 0.0。debug 脚本已更新。

## Step 3: 训练

| 参数 | 值 |
|------|-----|
| chunk_size | **50**（第一次 1 失败） |
| n_action_steps | 50 |
| dim_model | 128 |
| n_heads | 4 |
| n_encoder_layers | 2 |
| n_decoder_layers | 2 |
| use_vae | false |
| optimizer_lr | 3e-4 |
| steps | 10,000 |
| Params | 12.2M |
| Final loss | 0.058 |

**chunk_size=1 失败原因**: 无时序结构，模型在不同 phase 看到相同的 J2 位置却需要预测不同的动作（上升 vs 下降），只能坍缩到均值。chunk_size=50 让模型预测 50 步动作序列，提供了足够的时序上下文。

## Step 4: Teacher-Forcing Debug

| 指标 | Episode 0 | Episode 1 |
|------|-----------|-----------|
| Arm MSE | 0.0008 | 0.037 |
| improvement_ratio | **99.2%** | 67.9% |
| Close 时序误差 | **3 帧** | 4 帧 |
| Release 时序误差 | **1 帧** | 2 帧 |
| Gripper MSE | 8e-6 | 3.8e-4 |

所有 CHECK 通过，无坍缩。

## Step 5: Auto-Regressive Rollout

| 指标 | 结果 |
|------|------|
| Arm MSE | 0.015 |
| Gripper MSE | 0.00015 |
| Close 时序 | pred@100, true@106 (err=6) |
| Release 时序 | pred@213, true@228 (err=15) |
| J2 轨迹 0-150步 | Δ<0.03 rad |
| J2 轨迹 150-250步 | 漂移 ~0.1 rad |
| 终点 J2 | pred=0.000, true=0.006 |

模型成功复现了完整轨迹形状，夹爪时序在可接受范围内。中段 J2 漂移是正常的累积误差（12M 小模型 + 2 条轨迹）。

## 关键经验

1. **chunk_size ≥ 10 对 ACT 是必须的**，否则无时序结构导致坍缩
2. **夹爪阈值必须动态计算**（open_baseline vs grasp_width），不能假设闭合到 0.0
3. **debug 脚本需要传全部 image_features**（wrist + global），否则 KeyError
4. **auto-regressive rollout 才能暴露真实问题**，teacher-forcing 会掩盖累积误差

## 文件清单

| 文件 | 用途 |
|------|------|
| `teleop/data_collector.py` | 数据采集（--task-mode full_pick_place） |
| `training/train_act_full_fixed_overfit.sh` | Tiny ACT 训练脚本 |
| `scripts/debug_act_full_trajectory.py` | 离线 5 项 debug 检查（含动态阈值） |
| `inference/deploy.py` | 部署脚本（--policy-type act-full） |
| `outputs/train/act_full_fixed_overfit/checkpoints/010000/pretrained_model/` | 最终 checkpoint |

## 下一步（P1）

- 多位置采集（3-5 个不同 bottle 位置）
- 标准 ACT 配置（chunk_size=100, dim_model=512, use_vae=true）
- dry-run 离线自动回归测试（--dry-run，不需要 --allow-real-full-e2e）
- 真机分阶段验证（必须显式添加 --allow-real-full-e2e）
