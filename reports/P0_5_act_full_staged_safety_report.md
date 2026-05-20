# P0.5 act-full 分阶段安全验证准备报告

**日期**: 2026-05-19  
**状态**: 代码准备完成，等待真机分阶段验证  
**上一步**: P0 full trajectory ACT overfit — PASS

---

## 1. 背景

P0 full trajectory ACT overfit 已离线通过全部检查（teacher-forcing + auto-regressive rollout）。
Checkpoint：
```
outputs/train/act_full_fixed_overfit/checkpoints/010000/pretrained_model/
```

P0.5 的目标不是重新训练，而是为真机 full-e2e 增加**分阶段安全验证能力**。
直接一把梭完整 pick-and-place 过于危险 — 需要在 approach → close → lift → release 每个阶段都设置可提前停止的检查点。

## 2. 已完成修改

### 修改文件

| 文件 | 修改内容 |
|------|----------|
| `inference/deploy.py` | 新增 `--full-e2e-stop-after` 参数、分阶段停止逻辑、early close 检测、debug 日志 |
| `reports/P0_full_trajectory_act_report.md` | 修正 dry-run/real-run 说法 |

### 新增 CLI 参数

```
--full-e2e-stop-after {approach,close,lift,release,full}  (默认: approach)
```

该参数仅对 `--policy-type act-full` + `--test-mode full-e2e` 生效，其他 policy type 不受影响。

### 新增函数

| 函数 | 作用 |
|------|------|
| `get_gripper_phase()` | 基于动态中点 (grip_mid) 判断当前夹爪阶段：open / closing / closed / releasing / released |
| `check_phase_stop()` | 根据 `--full-e2e-stop-after` 模式判断是否应停止 |

### 默认行为

- `--full-e2e-stop-after approach` (默认) — 最保守，gripper 不闭合
- 没有 `--allow-real-full-e2e` 时强制 dry-run，拒绝真机执行

## 3. 分阶段停止逻辑

夹爪动态阈值（midpoint-based，适应瓶子宽度）：

```
grip_open = 0.08 (GRIPPER_OPEN)
grip_close = 0.0 (GRIPPER_CLOSE)
grip_mid = (grip_open + grip_close) / 2 = 0.04
```

### approach 停止条件
- gripper_pred 连续 3 帧低于 grip_mid → 即将闭合，立即停止
- 此时 gripper 尚未真正 close，确保 arm 已到达预抓取位置

### close 停止条件
- close_detected = True（gripper_pred 低于 grip_mid 持续 3 帧）→ 停止
- 用于检查闭合时机是否正确、是否夹到瓶子
- 停止后不 lift

### lift 停止条件
- close 后继续执行 30 帧 → 停止
- 用于检查是否夹稳、是否掉落
- 不移动到 place

### release 停止条件
- release_detected = True（close 后 gripper_pred 高于 grip_mid 持续 3 帧）→ 停止
- 用于检查放置前姿态和 release 时机

### full 执行
- 不提前停止，执行到 approach_steps（默认 200 步）或 ready_stop

### dry-run 局限
dry-run 不发机器人指令，robot_state 不变化，模型始终看到初始 observation。
因此 phase 检测无法在 dry-run 中真正推进到 close/release/lift，
只能在真机 closed-loop 下验证。

## 4. 安全机制

| 机制 | 状态 | 说明 |
|------|------|------|
| `--allow-real-full-e2e` 安全锁 | 已强制 | 无此参数时 act-full 只能 dry-run |
| Gripper clamp | 已生效 | gripper ∈ [0.0, 0.08]，不强制 OPEN |
| Early gripper close 检测 | 已生效 | 前 30 帧内 gripper 异常闭合 → 立即停止 |
| Per-joint max_delta | 已保留 | J1-J3: 0.03, J4-J6: 0.012 rad |
| EMA smoothing | 已保留 | alpha=0.5 |
| Joint limit violation | 已保留 | target 超出 ±3.0 rad → 停止 |
| Stagnation detection | 已保留 | dry-run 模式下跳过（robot 不动会误触发） |
| Wrist freeze | 已保留 | J2 > 1.45 时冻结 J4-J6 |
| Rollout save | 已保留 | --save-rollout 保存每帧数据 |
| Ctrl+C / Q 退出 | 已保留 | 安全断开电机 |

## 5. Dry-run 结果与局限

dry-run 已验证：
- CLI 参数解析正确
- Checkpoint 加载成功（12.2M params, chunk_size=50）
- act-full + full-e2e 入口正常
- `--allow-real-full-e2e` 安全锁正常
- Phase 跟踪变量初始化和日志打印正常
- 不会误发真机动作

dry-run 无法验证：
- 真实 phase progression（robot_state 不变化，模型始终预测 open phase）
- close/release/lift 的真实触发时机
- 瓶子是否被夹住

## 6. 真机命令模板

Checkpoint:
```
outputs/train/act_full_fixed_overfit/checkpoints/010000/pretrained_model/
```

### Stage 1: approach only
验证 arm 轨迹接近瓶子、gripper 不提前闭合：
```bash
python3 inference/deploy.py \
  --policy-type act-full \
  --checkpt outputs/train/act_full_fixed_overfit/checkpoints/010000/pretrained_model \
  --test-mode full-e2e \
  --full-e2e-stop-after approach \
  --allow-real-full-e2e \
  --debug-actions \
  --debug-policy-io \
  --save-rollout
```

### Stage 2: close only
验证 close 时机合理、不提前夹、不撞瓶：
```bash
python3 inference/deploy.py \
  --policy-type act-full \
  --checkpt outputs/train/act_full_fixed_overfit/checkpoints/010000/pretrained_model \
  --test-mode full-e2e \
  --full-e2e-stop-after close \
  --allow-real-full-e2e \
  --debug-actions \
  --debug-policy-io \
  --save-rollout
```

### Stage 3: lift
验证夹稳、只抬一点、不执行 place：
```bash
python3 inference/deploy.py \
  --policy-type act-full \
  --checkpt outputs/train/act_full_fixed_overfit/checkpoints/010000/pretrained_model \
  --test-mode full-e2e \
  --full-e2e-stop-after lift \
  --allow-real-full-e2e \
  --debug-actions \
  --debug-policy-io \
  --save-rollout
```

### Stage 4: release (可选)
验证放置前姿态和 release 时机：
```bash
python3 inference/deploy.py \
  --policy-type act-full \
  --checkpt outputs/train/act_full_fixed_overfit/checkpoints/010000/pretrained_model \
  --test-mode full-e2e \
  --full-e2e-stop-after release \
  --allow-real-full-e2e \
  --debug-actions \
  --debug-policy-io \
  --save-rollout
```

### Stage 5: full
全部通过后最后跑完整轨迹（必须有人监护急停）：
```bash
python3 inference/deploy.py \
  --policy-type act-full \
  --checkpt outputs/train/act_full_fixed_overfit/checkpoints/010000/pretrained_model \
  --test-mode full-e2e \
  --full-e2e-stop-after full \
  --allow-real-full-e2e \
  --debug-actions \
  --debug-policy-io \
  --save-rollout
```

> 以上命令为模板，本次没有执行。运行时必须显式添加 `--allow-real-full-e2e`。

## 7. 真机执行建议

### 推荐顺序

1. **Stage 1 approach** — arm 接近瓶子，gripper 保持 open
   - 观察 J2 是否接近 1.7 rad
   - 确认 gripper 没有提前闭合
   - 停止点应安全（不接触瓶子或仅轻轻接触）

2. **Stage 2 close** — approach 通过后再跑
   - 观察 close 帧是否在 100 附近
   - 确认夹爪是否真正夹到瓶子
   - 停止后检查瓶子是否稳固

3. **Stage 3 lift** — close 通过后再跑
   - 确认能夹住瓶子
   - 只抬一点（close + 30 帧）
   - 瓶子不应掉落

4. **Stage 4 release** (可选) — lift 通过后再跑
   - 检查放置前姿态
   - 检查 release 时机

5. **Stage 5 full** — 前三阶段全部通过后再跑
   - 必须有人准备急停（按 Q 或 Ctrl+C）
   - 完整 pick-and-place 轨迹

### 每次测试前检查
- 瓶子位置与采集时固定位置一致
- CAN 总线正常（can0）
- 手臂在安全起始姿态
- 有人监护

## 8. 不建议做的事

- 不要直接 full 一把梭，必须分阶段
- 不要跳过 approach → close → lift 顺序
- 不要用 diffusion checkpoint/dataset
- 不要混用 hybrid_v4_delta checkpoint
- 不要关闭安全机制（`--allow-real-full-e2e` 是必须的）
- 不要在瓶子位置与采集不一致时跑
- 不要无人值守运行

## 9. 当前结论

P0.5 代码准备完成。
- 分阶段停止逻辑（approach/close/lift/release/full）已实现
- 安全锁（`--allow-real-full-e2e`）已强制
- Dry-run 通过基础入口检查
- Dry-run 无法验证真实 phase progression（robot_state 不变化）

**下一步**：需要人工在真机上按 Stage 1 → Stage 2 → Stage 3 → Stage 4 → Stage 5 顺序分阶段测试。每阶段通过后再进入下一阶段。不要直接跑 full。
