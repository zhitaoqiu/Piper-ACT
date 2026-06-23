#!/usr/bin/env python3
"""
Piper SDK COMMAND node — SHADOW-WRITE (default) + single-step micro-motion.

Two modes:
  1. Shadow-only (default): subscribes to /piper/joint_states and
     /piper/command_joint_safe, performs secondary MAX_DELTA checks,
     prints would_send_target.  NO hardware writes.

  2. Single-step real write (--single-step-only): connects to SDK, enables
     the arm, sends exactly ONE joint offset, reads back, disables, exits.
     Requires --allow-real-write --confirm-real-write.

!!! SAFETY !!!
    Shadow-only mode NEVER imports the Piper SDK.
    Single-step mode imports PiperSdkAdapter ONLY after all gates pass.
"""

import argparse
import math
import sys
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from ros_bridge.common import JOINT_LIMITS, JOINT_NAMES, MAX_DELTA


# ---------------------------------------------------------------------------
#  Shadow-only subscription mode (unchanged from Phase 7)
# ---------------------------------------------------------------------------

class PiperSdkCommandNode(Node):
    def __init__(self, hz: float):
        super().__init__("piper_sdk_command_node")
        self._period = 1.0 / max(1.0, float(hz))
        self._current_qpos = None

        self._state_sub = self.create_subscription(
            JointState, "/piper/joint_states", self._on_joint_state, 10
        )
        self._cmd_sub = self.create_subscription(
            JointState, "/piper/command_joint_safe", self._on_command, 10
        )

        self._timer = self.create_timer(self._period, self._status_tick)

        self.get_logger().info("Piper SDK command node started in SHADOW-ONLY mode")
        self.get_logger().info("No SDK write commands will be sent in this phase")

    def _on_joint_state(self, msg: JointState):
        if len(msg.position) != 7:
            return
        self._current_qpos = [float(v) for v in msg.position]

    def _on_command(self, msg: JointState):
        if self._current_qpos is None:
            self.get_logger().warn("No current joint state yet, cannot validate command")
            return

        if list(msg.name) != JOINT_NAMES:
            self.get_logger().error(
                f"Joint name mismatch: expected {JOINT_NAMES}, got {list(msg.name)}"
            )
            return

        if len(msg.position) != 7:
            self.get_logger().error(
                f"Wrong position dim: expected 7, got {len(msg.position)}"
            )
            return

        for i, v in enumerate(msg.position):
            if math.isnan(v) or math.isinf(v):
                self.get_logger().error(f"Position[{i}] is NaN/Inf: {v}")
                return

        safe_target = [float(v) for v in msg.position]
        delta = [safe_target[i] - self._current_qpos[i] for i in range(7)]
        max_delta = [MAX_DELTA[name] for name in JOINT_NAMES]

        rejected = False
        for i in range(7):
            if abs(delta[i]) > max_delta[i] + 1e-6:
                self.get_logger().error(
                    f"REJECTED: {JOINT_NAMES[i]} delta {delta[i]:+.6f} "
                    f"exceeds MAX_DELTA {max_delta[i]:.6f}"
                )
                rejected = True

        if rejected:
            return

        self.get_logger().info(
            f"current_qpos:    {[f'{v:.6f}' for v in self._current_qpos]}"
        )
        self.get_logger().info(
            f"safe_target:      {[f'{v:.6f}' for v in safe_target]}"
        )
        self.get_logger().info(
            f"delta:            {[f'{v:+.6f}' for v in delta]}"
        )
        self.get_logger().info(
            f"max_delta:        {[f'{v:.6f}' for v in max_delta]}"
        )
        self.get_logger().info("ACCEPTED")
        self.get_logger().info(
            f"would_send_target: {[f'{v:.6f}' for v in safe_target]}"
        )

    def _status_tick(self):
        if self._current_qpos is not None:
            self.get_logger().info(
                f"qpos: {[f'{v:.4f}' for v in self._current_qpos]}",
                throttle_duration_sec=5.0,
            )

    def destroy_node(self):
        super().destroy_node()


# ---------------------------------------------------------------------------
#  Single-step real write (Phase 8 — micro-motion test)
# ---------------------------------------------------------------------------

def _run_single_step(parsed: argparse.Namespace) -> int:
    # --- gate 1: require both confirm flags --------------------------------
    if not parsed.allow_real_write or not parsed.confirm_real_write:
        print(
            "\n=============================================================="
            "\n  ERROR: --single-step-only requires both --allow-real-write"
            "\n  and --confirm-real-write.  Aborting."
            "\n==============================================================\n"
        )
        return 1

    # --- gate 2: validate joint name ---------------------------------------
    if parsed.joint_name not in JOINT_NAMES:
        print(f"ERROR: unknown joint '{parsed.joint_name}'. Must be one of {JOINT_NAMES}")
        return 1
    joint_idx = JOINT_NAMES.index(parsed.joint_name)

    # --- gate 3: validate delta --------------------------------------------
    delta_val = float(parsed.delta)
    max_real = float(parsed.max_real_delta)
    if abs(delta_val) > max_real + 1e-9:
        print(
            f"ERROR: delta {delta_val:+.6f} exceeds --max-real-delta {max_real:.6f}"
        )
        return 1

    print(
        "\n##############################################################"
        "\n#  SINGLE-STEP REAL WRITE"
        f"\n#  Joint: {parsed.joint_name}   Delta: {delta_val:+.6f} rad"
        f"\n#  Max allowed: {max_real:.6f} rad"
        "\n#"
        "\n#  This will ENABLE the arm, send ONE position command,"
        "\n#  read back the result, DISABLE the arm, and EXIT."
        "\n##############################################################\n"
    )

    # --- step 1: get "before" from state_node via ROS (warm connection) ----
    rclpy.init(args=sys.argv)
    node = rclpy.create_node("piper_sdk_single_step")
    before = None
    after = None

    def _on_js(msg: JointState):
        nonlocal before, after
        if len(msg.position) != 7:
            return
        qpos = [float(v) for v in msg.position]
        if before is None:
            before = qpos
        after = qpos  # always keep latest

    sub = node.create_subscription(JointState, "/piper/joint_states", _on_js, 10)
    print("Waiting for /piper/joint_states ...")

    deadline = time.monotonic() + 5.0
    while before is None and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)

    if before is None:
        print("ERROR: no /piper/joint_states received within 5 s. "
              "Is piper_sdk_state_node running?")
        node.destroy_node()
        rclpy.shutdown()
        return 1

    print(f"Before qpos:     {[f'{v:.6f}' for v in before]}")
    after = list(before)  # use before as fallback

    # --- step 2: import SDK, connect (for write only) -----------------------
    print("Importing Piper SDK adapter ...")
    from piper_sdk_py_driver.sdk_adapter import PiperSdkAdapter

    adapter = PiperSdkAdapter(
        can_port=parsed.can_port,
        gripper_exist=True,
        enable_timeout=10.0,
    )
    adapter.connect()
    print(f"Connected to {parsed.can_port}")

    # --- step 2: calculate target ------------------------------------------
    target = list(before)
    target[joint_idx] += delta_val

    # --- gate 4: joint limits check (arm joints only; gripper untouched) ---
    for i, name in enumerate(JOINT_NAMES):
        if name == "gripper":
            continue
        lo, hi = JOINT_LIMITS.get(name, (-math.inf, math.inf))
        if target[i] < lo - 1e-9 or target[i] > hi + 1e-9:
            print(f"ERROR: target[{name}] = {target[i]:.6f} outside limit [{lo}, {hi}]")
            adapter.disconnect()
            node.destroy_node()
            rclpy.shutdown()
            return 1

    # --- gate 5: MAX_DELTA check -------------------------------------------
    for i, name in enumerate(JOINT_NAMES):
        d = target[i] - before[i]
        limit = MAX_DELTA.get(name, 0.02)
        if abs(d) > limit + 1e-6:
            print(f"ERROR: {name} delta {d:+.6f} exceeds MAX_DELTA {limit:.6f}")
            adapter.disconnect()
            node.destroy_node()
            rclpy.shutdown()
            return 1

    # --- gate 6: only the named joint should change (warning, not error) ---
    for i, name in enumerate(JOINT_NAMES):
        d = abs(target[i] - before[i])
        if d > 1e-9 and name != parsed.joint_name:
            print(f"WARNING: {name} has nonzero delta {target[i]-before[i]:+.6f}")

    # --- step 3: enable (skip gripper init to preserve position) ---------------
    print("Enabling arm (blocking, enable_gripper=False) ...")
    ok = adapter.enable(blocking=True, enable_gripper=False)
    if not ok:
        print("ERROR: arm enable failed (timeout or driver not ready)")
        adapter.disconnect()
        node.destroy_node()
        rclpy.shutdown()
        return 1
    print("Arm enabled")

    # --- step 4: send ONE command (arm joints only — no GripperCtrl) ----------
    print(f"Sending: {parsed.joint_name} target {target[joint_idx]:.6f} "
          f"(delta {delta_val:+.6f}) ...")
    # Pass only 6 arm joints so send_joint_positions never calls GripperCtrl.
    # Gripper is preserved passively — no CAN frame touches it on this path.
    send_time = time.monotonic()
    adapter.send_joint_positions(target[:6], velocity_percent=30)
    print("Command sent.")

    # -------------------------------------------------------------------
    #  DIAGNOSTIC PATH: multi-sample reads while arm stays enabled
    # -------------------------------------------------------------------
    if parsed.diagnose_enabled_read:
        samples = {}
        read_times = {}

        for label, delay in [("after_0p1s", 0.1), ("after_0p3s", 0.3),
                             ("after_0p5s", 0.5), ("after_1p0s", 1.0)]:
            remaining = send_time + delay - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            try:
                js = adapter.read_joint_state()
                samples[label] = [float(v) for v in js.position]
                read_times[label] = time.monotonic() - send_time
            except Exception as exc:
                print(f"WARNING: SDK read at {label} failed: {exc}")
                samples[label] = [float('nan')] * 7
                read_times[label] = time.monotonic() - send_time

        # --- read state_node while arm is STILL enabled (no disable) ---------
        print("Reading state_node while arm enabled ...")
        time.sleep(0.2)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        for _ in range(10):
            rclpy.spin_once(node, timeout_sec=0.05)
        samples["after_1p0s_sn"] = (
            list(after) if len(after) == 7 else [float('nan')] * 7
        )

        # --- disconnect WITHOUT disabling arm --------------------------------
        # Bypass DisableArm in adapter.disconnect() so arm holds position.
        adapter._enabled = False
        adapter.disconnect()
        print("Disconnected from CAN (arm left ENABLED)")
        node.destroy_node()
        rclpy.shutdown()

        # --- print diagnostic table ------------------------------------------
        print(
            "\n=============================================================="
            "\n  ENABLED-WINDOW DIAGNOSTIC RESULT  (arm stays ENABLED)"
            "\n=============================================================="
        )
        print("  SDK reads (after_0p1s..after_1p0s): direct adapter — may show")
        print("  CAN corruption on J2/J3 (converge to 0.0 = artifact).")
        print("  after_1p0s_sn: state_node via ROS while arm still enabled.")
        header = f"{'Sample':>16}"
        for name in JOINT_NAMES:
            header += f" {name:>12}"
        header += f" {'J2_delta':>12} {'J5_delta':>12}"
        print(header)
        print("-" * len(header))

        row_order = ["before", "after_0p1s", "after_0p3s", "after_0p5s",
                     "after_1p0s", "after_1p0s_sn"]
        all_samples = {"before": before, **samples}
        for label in row_order:
            vals = all_samples.get(label)
            if vals is None or len(vals) < 7:
                continue
            row = f"{label:>16}"
            for v in vals:
                row += f" {'nan':>12}" if math.isnan(v) else f" {v:12.6f}"
            j2_d = vals[1] - before[1]
            j5_d = vals[4] - before[4]
            row += f" {'nan':>12}" if math.isnan(vals[1]) else f" {j2_d:+12.6f}"
            row += f" {'nan':>12}" if math.isnan(vals[4]) else f" {j5_d:+12.6f}"
            print(row)

        # --- summary ---------------------------------------------------------
        print()
        after_1p0s = samples.get("after_1p0s", [float('nan')] * 7)
        after_sn = samples.get("after_1p0s_sn", [float('nan')] * 7)
        ok_1p0s = len(after_1p0s) == 7 and not any(math.isnan(v) for v in after_1p0s)
        ok_sn = len(after_sn) == 7 and not any(math.isnan(v) for v in after_sn)

        # State-node reading is the ground truth (not corrupted by cmd SDK).
        if ok_sn:
            j2_sn = after_sn[1] - before[1]
            j5_sn = after_sn[4] - before[4]
            print(f"joint2 delta (state_node, enabled): requested {delta_val:+.6f}  actual {j2_sn:+.6f}")
            print(f"joint5 delta (state_node, enabled): {j5_sn:+.6f}")
            if abs(j2_sn - delta_val) < 0.001:
                print("=> joint2 reached target while enabled (state_node confirms)")
            else:
                print(f"=> joint2 did NOT reach target (off by {abs(j2_sn - delta_val):.6f} rad)")
                print("   Possible causes: joint index mapping in SDK, or arm controller")

        if ok_1p0s:
            j2_act = after_1p0s[1] - before[1]
            j5_act = after_1p0s[4] - before[4]
            print(f"joint2 delta at +1.0s (SDK direct): {j2_act:+.6f}  [CORRUPTED if near 0]")
            print(f"joint5 delta at +1.0s (SDK direct): {j5_act:+.6f}")

        # --- gripper stability -----------------------------------------------
        gripper_ok = True
        for label in row_order:
            vals = all_samples.get(label)
            if vals is None or len(vals) != 7 or math.isnan(vals[6]):
                print(f"WARNING: gripper reading invalid at {label}")
                gripper_ok = False
                continue
            d = vals[6] - before[6]
            if abs(d) > 0.001:
                print(f"WARNING: gripper moved at {label}: delta {d:+.6f} m")
                gripper_ok = False
        if gripper_ok:
            print("gripper: stable (all deltas < 0.001 m)")
        print("\n  Arm is still ENABLED — disable manually when ready.")

        return 0

    # -------------------------------------------------------------------
    #  NORMAL PATH: settle, disable, read after, report
    # -------------------------------------------------------------------
    print("Settling (arm enabled) ...")
    time.sleep(1.0)

    # --- step 5: disable arm, THEN read after (telemetry clean) ------------
    adapter.disable()
    print("Arm disabled. Reading position ...")
    time.sleep(0.3)

    # Read from state_node's warm connection — the CAN bus is quiet now
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    for _ in range(10):
        rclpy.spin_once(node, timeout_sec=0.05)
    print(f"After qpos:      {[f'{v:.6f}' for v in after]}")

    # --- step 6: disconnect + ROS cleanup -----------------------------------
    adapter.disconnect()
    print("Disconnected from CAN")
    node.destroy_node()
    rclpy.shutdown()
    print(
        "\n=============================================================="
        "\n  SINGLE-STEP RESULT"
        "\n=============================================================="
    )
    header = f"{'Joint':>10} {'before':>12} {'target':>12} {'after':>12} {'delta':>12} {'error':>12}"
    print(header)
    print("-" * len(header))
    for i, name in enumerate(JOINT_NAMES):
        b = before[i]
        t = target[i]
        a = after[i]
        d = t - b
        e = a - b
        marker = " ***" if abs(t - b) > 1e-9 else ""
        print(f"{name:>10} {b:12.6f} {t:12.6f} {a:12.6f} {d:+12.6f}{marker} {e:+12.6f}")

    actual_delta = after[joint_idx] - before[joint_idx]
    print(f"\n{parsed.joint_name} delta: requested {delta_val:+.6f}  actual {actual_delta:+.6f}")

    return 0


# ---------------------------------------------------------------------------
#  Session trajectory test (Phase 9 — enabled-session short trajectory)
# ---------------------------------------------------------------------------

def _run_session_trajectory_test(parsed: argparse.Namespace) -> int:
    # --- gate 1: require both confirm flags --------------------------------
    if not parsed.allow_real_write or not parsed.confirm_real_write:
        print(
            "\n=============================================================="
            "\n  ERROR: --session-trajectory-test requires both"
            "\n  --allow-real-write and --confirm-real-write.  Aborting."
            "\n==============================================================\n"
        )
        return 1

    # --- gate 2: validate joint name (arm joints only, no gripper) ----------
    arm_joints = JOINT_NAMES[:6]
    if parsed.joint_name not in arm_joints:
        print(f"ERROR: --joint-name must be one of {arm_joints}, "
              f"got '{parsed.joint_name}'")
        return 1
    joint_idx = arm_joints.index(parsed.joint_name)

    # --- gate 3: total delta check ------------------------------------------
    step_delta = float(parsed.step_delta)
    num_steps = int(parsed.num_steps)
    max_total = float(parsed.max_total_delta)
    total_delta = step_delta * num_steps
    if abs(total_delta) > max_total + 1e-9:
        print(
            f"ERROR: total delta {total_delta:+.6f} rad ({num_steps} steps x "
            f"{step_delta:+.6f}) exceeds --max-total-delta {max_total:.6f}"
        )
        return 1

    rate = float(parsed.rate)
    if rate > 2.0:
        print(f"ERROR: --rate {rate} exceeds max 2 Hz for this phase")
        return 1
    period = 1.0 / max(0.5, rate)

    print(
        "\n##############################################################"
        "\n#  SESSION TRAJECTORY TEST"
        f"\n#  Joint: {parsed.joint_name}   Steps: {num_steps}"
        f"\n#  Step delta: {step_delta:+.6f} rad   Rate: {rate} Hz"
        f"\n#  Total delta: {total_delta:+.6f} rad   Max: {max_total:.6f}"
        "\n#"
        "\n#  Enable ONCE, send trajectory, read each step, exit."
        "\n#  Arm stays ENABLED — no GripperCtrl calls."
        "\n##############################################################\n"
    )

    # --- step 1: get "before" from state_node via ROS -----------------------
    rclpy.init(args=sys.argv)
    node = rclpy.create_node("piper_sdk_trajectory_test")
    before = None

    def _on_js(msg: JointState):
        nonlocal before
        if len(msg.position) != 7:
            return
        if before is None:
            before = [float(v) for v in msg.position]

    sub = node.create_subscription(JointState, "/piper/joint_states", _on_js, 10)
    print("Waiting for /piper/joint_states ...")

    deadline = time.monotonic() + 5.0
    while before is None and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)

    if before is None:
        print("ERROR: no /piper/joint_states received within 5 s. "
              "Is piper_sdk_state_node running?")
        node.destroy_node()
        rclpy.shutdown()
        return 1

    print(f"Before qpos (state_node): {[f'{v:.6f}' for v in before]}")

    # --- step 2: import SDK, connect ----------------------------------------
    print("Importing Piper SDK adapter ...")
    from piper_sdk_py_driver.sdk_adapter import PiperSdkAdapter

    adapter = PiperSdkAdapter(
        can_port=parsed.can_port,
        gripper_exist=True,
        enable_timeout=10.0,
    )
    adapter.connect()
    print(f"Connected to {parsed.can_port}")

    # --- step 3: enable (once) ----------------------------------------------
    print("Enabling arm (blocking, enable_gripper=False) ...")
    ok = adapter.enable(blocking=True, enable_gripper=False)
    if not ok:
        print("ERROR: arm enable failed (timeout or driver not ready)")
        adapter.disconnect()
        node.destroy_node()
        rclpy.shutdown()
        return 1
    print("Arm enabled. Waiting 0.8 s for settle ...")
    time.sleep(0.8)

    # --- step 4: read q0 (stable enabled position) --------------------------
    try:
        js = adapter.read_joint_state()
        q0 = [float(v) for v in js.position]
    except Exception as exc:
        print(f"ERROR: q0 read failed: {exc}")
        adapter._enabled = False
        adapter.disconnect()
        node.destroy_node()
        rclpy.shutdown()
        return 1
    print(f"q0 (enabled, stable):  {[f'{v:.6f}' for v in q0]}")

    # --- step 5: build trajectory targets -----------------------------------
    targets = []
    for step in range(1, num_steps + 1):
        t = list(q0)
        t[joint_idx] = q0[joint_idx] + step_delta * step
        targets.append(t)

    # --- gate 4: validate all targets against JOINT_LIMITS ------------------
    for step, t in enumerate(targets, 1):
        for i, name in enumerate(JOINT_NAMES):
            if name == "gripper":
                continue
            lo, hi = JOINT_LIMITS.get(name, (-math.inf, math.inf))
            if t[i] < lo - 1e-9 or t[i] > hi + 1e-9:
                print(f"ERROR: step {step} target[{name}] = {t[i]:.6f} "
                      f"outside limit [{lo}, {hi}]")
                adapter._enabled = False
                adapter.disconnect()
                node.destroy_node()
                rclpy.shutdown()
                return 1

    # --- gate 5: validate per-step MAX_DELTA --------------------------------
    for step, t in enumerate(targets, 1):
        prev = q0 if step == 1 else targets[step - 2]
        for i, name in enumerate(JOINT_NAMES):
            d = t[i] - prev[i]
            limit = MAX_DELTA.get(name, 0.02)
            if abs(d) > limit + 1e-6:
                print(f"ERROR: step {step} {name} delta {d:+.6f} "
                      f"exceeds MAX_DELTA {limit:.6f}")
                adapter._enabled = False
                adapter.disconnect()
                node.destroy_node()
                rclpy.shutdown()
                return 1

    # --- step 6: execute trajectory -----------------------------------------
    print(f"\nExecuting trajectory: {num_steps} steps at {rate} Hz ...")
    rows = []

    for step, t in enumerate(targets, 1):
        adapter.send_joint_positions(t[:6], velocity_percent=30)
        time.sleep(period)

        try:
            js = adapter.read_joint_state()
            actual = [float(v) for v in js.position]
        except Exception as exc:
            print(f"WARNING: read at step {step} failed: {exc}")
            actual = [float('nan')] * 7

        rows.append((
            step, t[joint_idx], actual[joint_idx],
            actual[joint_idx] - q0[joint_idx],
            actual[joint_idx] - t[joint_idx],
            actual[0], actual[2], actual[3], actual[4], actual[5], actual[6],
        ))

    # --- step 7: cleanup (keep arm enabled — consistent with diagnostic) ----
    adapter._enabled = False
    adapter.disconnect()
    print("Disconnected from CAN (arm left ENABLED)")
    node.destroy_node()
    rclpy.shutdown()

    # --- print results table ------------------------------------------------
    print(
        "\n=============================================================="
        "\n  SESSION TRAJECTORY RESULT"
        "\n=============================================================="
    )
    header = (f"{'Step':>5} {'target_j2':>12} {'actual_j2':>12} "
              f"{'d_from_q0':>12} {'error':>12} "
              f"{'j1':>10} {'j3':>10} {'j4':>10} {'j5':>10} {'j6':>10} {'grip':>10}")
    print(header)
    print("-" * len(header))

    for row in rows:
        step, tj2, aj2, dq0, err, j1, j3, j4, j5, j6, gr = row
        def _f(v):
            return f"{'nan':>10}" if math.isnan(v) else f"{v:10.6f}"
        print(f"{step:>5} {tj2:12.6f} {aj2:12.6f} {dq0:+12.6f} {err:+12.6f} "
              f"{_f(j1)} {_f(j3)} {_f(j4)} {_f(j5)} {_f(j6)} {_f(gr)}")

    # --- summary ------------------------------------------------------------
    print()
    print(f"q0 joint2: {q0[joint_idx]:.6f}")
    print(f"Target trajectory: {[f'{t[joint_idx]:.6f}' for t in targets]}")
    print(f"Actual trajectory: {[f'{r[2]:.6f}' for r in rows]}")
    print(f"Total requested: {total_delta:+.6f} rad")
    if rows and not any(math.isnan(r[2]) for r in rows):
        total_actual = rows[-1][2] - q0[joint_idx]
        print(f"Total actual:    {total_actual:+.6f} rad")

        # monotonicity
        actuals = [r[2] for r in rows]
        mono = all(actuals[i] <= actuals[i+1] + 1e-9 for i in range(len(actuals)-1))
        print(f"Monotonic: {'YES' if mono else 'NO'}")

        # other-joint stability
        def _max_d(name_idx):
            return max(abs(r[5 + name_idx] - q0[name_idx + (0 if name_idx < 1 else 1 if name_idx < 2 else 2)])
                       for r in rows) if rows else 0
        # j1=idx0, j3=idx2, j4=idx3, j5=idx4, j6=idx5 in rows tuple
        max_j1 = max(abs(r[5] - q0[0]) for r in rows)
        max_j3 = max(abs(r[6] - q0[2]) for r in rows)
        max_j4 = max(abs(r[7] - q0[3]) for r in rows)
        max_j5 = max(abs(r[8] - q0[4]) for r in rows)
        max_j6 = max(abs(r[9] - q0[5]) for r in rows)
        print(f"Max other-joint drift: j1={max_j1:.6f} j3={max_j3:.6f} "
              f"j4={max_j4:.6f} j5={max_j5:.6f} j6={max_j6:.6f}")

        max_grip = max(abs(r[10] - q0[6]) for r in rows)
        print(f"Max gripper drift: {max_grip:.6f} m {'OK' if max_grip < 0.001 else 'WARNING'}")

    print("\n  Arm is still ENABLED — disable manually when ready.")
    return 0


# ---------------------------------------------------------------------------
#  ACT real-write session (Phase 10B — 20%-scale micro-motion)
# ---------------------------------------------------------------------------

def _run_real_write_session(parsed: argparse.Namespace) -> int:
    # --- gate 1: require both confirm flags ----------------------------------
    if not parsed.allow_real_write or not parsed.confirm_real_write:
        print(
            "\n=============================================================="
            "\n  ERROR: --real-write-session requires both --allow-real-write"
            "\n  and --confirm-real-write.  Aborting."
            "\n==============================================================\n"
        )
        return 1

    command_scale = float(parsed.command_scale)
    if not 0.0 < command_scale <= 1.0:
        print(f"ERROR: --command-scale must be in (0.0, 1.0], got {command_scale}")
        return 1

    max_write_steps = int(parsed.max_write_steps)
    rate = float(parsed.rate)
    if rate > 2.0:
        print(f"ERROR: --rate {rate} exceeds max 2 Hz for Phase 10B")
        return 1
    min_interval = 1.0 / rate
    freeze_gripper = parsed.freeze_gripper
    max_cumul_j2 = float(parsed.max_cumulative_delta_j2)
    max_cumul_any = float(parsed.max_cumulative_delta_any_joint)

    print(
        "\n##############################################################"
        "\n#  ACT REAL-WRITE SESSION (Phase 10F)"
        f"\n#  Command scale: {command_scale:.1%}   Max steps: {max_write_steps}"
        f"\n#  Rate limit: {rate} Hz   Freeze gripper: {freeze_gripper}"
        f"\n#  Cumulative guard J2: {max_cumul_j2:.4f}  Any joint: {max_cumul_any:.4f}"
        "\n#"
        "\n#  ENABLE once, process safe commands from /piper/command_joint_safe,"
        f"\n#  write scaled targets (arm joints only, no gripper), exit."
        "\n##############################################################\n"
    )

    # --- init rclpy + shared state -------------------------------------------
    rclpy.init(args=sys.argv)
    node = rclpy.create_node("piper_sdk_real_write_session")
    node.get_logger().info("Real-write session node created")

    current_qpos = None
    qpos_timestamp = 0.0
    write_count = 0
    done = False
    error_exit = False
    gripper_initial = None
    session_start_qpos = None
    last_write_time = 0.0
    step_log = []
    raw_target_log = []  # (step, raw_gripper) for Phase 13+ gripper shadow
    latest_raw_target = None  # most recent /policy/target_joint_raw

    def _on_js(msg: JointState):
        nonlocal current_qpos, qpos_timestamp
        if len(msg.position) != 7:
            return
        current_qpos = [float(v) for v in msg.position]
        qpos_timestamp = time.monotonic()

    def _on_raw(msg: JointState):
        nonlocal latest_raw_target
        if len(msg.position) == 7:
            latest_raw_target = [float(v) for v in msg.position]

    def _on_command(msg: JointState):
        nonlocal write_count, done, error_exit, current_qpos, qpos_timestamp
        nonlocal gripper_initial, session_start_qpos, last_write_time, step_log
        nonlocal raw_target_log, latest_raw_target

        if done:
            return

        if write_count >= max_write_steps:
            return

        # --- rate limit -------------------------------------------------------
        now = time.monotonic()
        if last_write_time > 0.0 and (now - last_write_time) < min_interval - 0.005:
            return  # skip — too soon

        # --- state freshness check --------------------------------------------
        if current_qpos is None:
            node.get_logger().warn("No joint state yet, skipping command")
            return

        age = now - qpos_timestamp
        if age > 0.5:
            node.get_logger().error(
                f"Joint state STALE ({age:.3f}s > 0.5s). REJECTED. Stopping session."
            )
            done = True
            error_exit = True
            return

        # --- validate safe target ---------------------------------------------
        if list(msg.name) != JOINT_NAMES:
            node.get_logger().error(
                f"Joint name mismatch: expected {JOINT_NAMES}, got {list(msg.name)}. Stopping."
            )
            done = True
            error_exit = True
            return

        if len(msg.position) != 7:
            node.get_logger().error(f"Wrong position dim: {len(msg.position)}. Stopping.")
            done = True
            error_exit = True
            return

        for i, v in enumerate(msg.position):
            if math.isnan(v) or math.isinf(v):
                node.get_logger().error(f"Position[{i}] NaN/Inf: {v}. Stopping.")
                done = True
                error_exit = True
                return

        safe_target = [float(v) for v in msg.position]
        delta = [safe_target[i] - current_qpos[i] for i in range(7)]

        # --- secondary MAX_DELTA check ----------------------------------------
        rejected = False
        for i in range(7):
            limit = MAX_DELTA.get(JOINT_NAMES[i], 0.02)
            if abs(delta[i]) > limit + 1e-6:
                node.get_logger().error(
                    f"REJECTED: {JOINT_NAMES[i]} delta {delta[i]:+.6f} "
                    f"exceeds MAX_DELTA {limit:.6f}"
                )
                rejected = True
        if rejected:
            node.get_logger().error("MAX_DELTA violation — stopping session.")
            done = True
            error_exit = True
            return

        # --- compute scaled target --------------------------------------------
        target_scaled = [current_qpos[i] + command_scale * delta[i] for i in range(7)]

        # freeze gripper at current position
        if freeze_gripper:
            target_scaled[6] = current_qpos[6]

        # --- re-check scaled delta --------------------------------------------
        scaled_delta = [target_scaled[i] - current_qpos[i] for i in range(7)]
        for i in range(7):
            limit = MAX_DELTA.get(JOINT_NAMES[i], 0.02)
            if abs(scaled_delta[i]) > limit + 1e-6:
                node.get_logger().error(
                    f"SCALED REJECTED: {JOINT_NAMES[i]} scaled_delta "
                    f"{scaled_delta[i]:+.6f} > {limit:.6f}. Stopping."
                )
                done = True
                error_exit = True
                return

        # --- capture gripper initial on first write ---------------------------
        if gripper_initial is None:
            gripper_initial = current_qpos[6]

        # --- SEND (arm joints only — no GripperCtrl) --------------------------
        write_count += 1
        last_write_time = now
        adapter.send_joint_positions(target_scaled[:6], velocity_percent=30)

        # --- read after-state (brief settle, then SDK read) -------------------
        time.sleep(0.15)
        try:
            js = adapter.read_joint_state()
            after_state = [float(v) for v in js.position]
        except Exception:
            after_state = [float('nan')] * 7

        # --- spin for updated state_node reading ------------------------------
        for _ in range(5):
            rclpy.spin_once(node, timeout_sec=0.05)
        after_sn = current_qpos  # updated by _on_js via spin

        # --- gripper movement check -------------------------------------------
        gripper_delta = (after_sn[6] if after_sn is not None else float('nan')) - gripper_initial
        if not math.isnan(gripper_delta) and abs(gripper_delta) > 0.001:
            node.get_logger().error(
                f"GRIPPER MOVED: delta {gripper_delta:+.6f}m > 0.001m. Stopping session."
            )
            done = True
            error_exit = True
            step_log.append((
                write_count, list(current_qpos) if current_qpos else [float('nan')]*7,
                safe_target, delta, target_scaled, after_state, after_sn
            ))
            return

        # --- capture session_start_qpos on first write -------------------------
        if session_start_qpos is None:
            session_start_qpos = list(current_qpos) if current_qpos else None

        # --- cumulative delta guard (from session start) -----------------------
        if session_start_qpos is not None and after_sn is not None:
            cumul_delta = [after_sn[i] - session_start_qpos[i] for i in range(7)]
            # J2 guard
            if abs(cumul_delta[2]) > max_cumul_j2:
                node.get_logger().error(
                    f"CUMULATIVE GUARD: J2 cumulative delta {cumul_delta[2]:+.6f} "
                    f"exceeds limit {max_cumul_j2:.4f}. Stopping session."
                )
                done = True
                error_exit = True
                step_log.append((
                    write_count, list(current_qpos) if current_qpos else [float('nan')]*7,
                    safe_target, delta, target_scaled, after_state, after_sn
                ))
                return
            # any-arm-joint guard
            max_arm_cumul = max(abs(cumul_delta[i]) for i in range(6))
            if max_arm_cumul > max_cumul_any:
                worst_idx = max(range(6), key=lambda i: abs(cumul_delta[i]))
                node.get_logger().error(
                    f"CUMULATIVE GUARD: {JOINT_NAMES[worst_idx]} cumulative delta "
                    f"{cumul_delta[worst_idx]:+.6f} exceeds limit {max_cumul_any:.4f}. Stopping session."
                )
                done = True
                error_exit = True
                step_log.append((
                    write_count, list(current_qpos) if current_qpos else [float('nan')]*7,
                    safe_target, delta, target_scaled, after_state, after_sn
                ))
                return

        # --- oscillation / single-step guard (Phase 12+) -----------------------
        if write_count >= 2 and len(step_log) >= 1:
            prev_before = step_log[-1][1]  # current_qpos from previous step
            curr_after = after_sn if after_sn is not None else (list(current_qpos) if current_qpos else None)
            if prev_before is not None and curr_after is not None:
                j2_step = abs(curr_after[2] - prev_before[2])
                j3_step = abs(curr_after[3] - prev_before[3])
                if j2_step > MAX_DELTA["joint2"] + 0.003:
                    node.get_logger().error(
                        f"OSCILLATION GUARD: J2 single-step delta {j2_step:.6f} > {MAX_DELTA['joint2'] + 0.003}. Stopping."
                    )
                    done = True
                    error_exit = True
                    step_log.append((
                        write_count, list(current_qpos) if current_qpos else [float('nan')]*7,
                        safe_target, delta, target_scaled, after_state, after_sn
                    ))
                    return
                if j3_step > MAX_DELTA["joint3"] + 0.003:
                    node.get_logger().error(
                        f"OSCILLATION GUARD: J3 single-step delta {j3_step:.6f} > {MAX_DELTA['joint3'] + 0.003}. Stopping."
                    )
                    done = True
                    error_exit = True
                    step_log.append((
                        write_count, list(current_qpos) if current_qpos else [float('nan')]*7,
                        safe_target, delta, target_scaled, after_state, after_sn
                    ))
                    return
            # sliding-window oscillation check: last 6 steps
            if len(step_log) >= 5:
                recent = [step_log[i][1] for i in range(-5, 0)] + [list(current_qpos) if current_qpos else None]  # last 5 logged + current
                if all(b is not None for b in recent):
                    j2_vals = [b[2] for b in recent]
                    j2_net = abs(j2_vals[-1] - j2_vals[0])
                    j2_abs = sum(abs(j2_vals[i+1] - j2_vals[i]) for i in range(5))
                    if j2_net < 0.005 and j2_abs > 0.04:
                        node.get_logger().error(
                            f"OSCILLATION GUARD: J2 6-step net={j2_net:.5f} < 0.005 "
                            f"but abs={j2_abs:.5f} > 0.04 (ineffective oscillation). Stopping."
                        )
                        done = True
                        error_exit = True
                        step_log.append((
                            write_count, list(current_qpos) if current_qpos else [float('nan')]*7,
                            safe_target, delta, target_scaled, after_state, after_sn
                        ))
                        return

        # --- log step ---------------------------------------------------------
        raw_grip = latest_raw_target[6] if latest_raw_target is not None and len(latest_raw_target) == 7 else float('nan')
        raw_target_log.append((write_count, raw_grip, safe_target[6], current_qpos[6] if current_qpos else float('nan')))
        step_log.append((
            write_count,
            list(current_qpos) if current_qpos else [float('nan')]*7,
            safe_target, delta, target_scaled, after_state,
            list(after_sn) if after_sn else [float('nan')]*7,
        ))

        node.get_logger().info(
            f"--- Step {write_count}/{max_write_steps} ---"
        )
        node.get_logger().info(
            f"current_qpos:     {[f'{v:.6f}' for v in (current_qpos if current_qpos else [0.0]*7)]}"
        )
        node.get_logger().info(
            f"safe_target:      {[f'{v:.6f}' for v in safe_target]}"
        )
        node.get_logger().info(
            f"raw gripper: {raw_grip:.5f}  safe gripper: {safe_target[6]:.5f}  actual: {current_qpos[6]:.5f}"
            + ("  [frozen]" if freeze_gripper else "")
        )
        node.get_logger().info(
            f"raw delta:        {[f'{v:+.6f}' for v in delta]}"
        )
        node.get_logger().info(
            f"target_scaled:    {[f'{v:.6f}' for v in target_scaled]}"
        )
        node.get_logger().info(
            f"scaled_delta:     {[f'{v:+.6f}' for v in scaled_delta]}"
        )
        node.get_logger().info(
            f"after (SDK):      {[f'{v:.6f}' for v in after_state]}"
        )
        node.get_logger().info(
            f"after (state_node):{[f'{v:.6f}' for v in (after_sn if after_sn else [0.0]*7)]}"
        )
        node.get_logger().info("ACCEPTED + WRITTEN")

        # --- check max steps --------------------------------------------------
        if write_count >= max_write_steps:
            node.get_logger().info(
                f"Reached max write steps ({max_write_steps}). Finishing session."
            )
            done = True

    # --- subscribe ------------------------------------------------------------
    node.create_subscription(JointState, "/piper/joint_states", _on_js, 10)
    node.create_subscription(JointState, "/piper/command_joint_safe", _on_command, 10)
    node.create_subscription(JointState, "/policy/target_joint_raw", _on_raw, 10)

    # --- wait for first joint state -------------------------------------------
    print("Waiting for /piper/joint_states ...")
    deadline = time.monotonic() + 5.0
    while current_qpos is None and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)

    if current_qpos is None:
        print("ERROR: no /piper/joint_states received within 5 s. "
              "Is piper_sdk_state_node running?")
        node.destroy_node()
        rclpy.shutdown()
        return 1

    node.get_logger().info(
        f"Initial qpos: {[f'{v:.6f}' for v in current_qpos]}"
    )

    # --- import SDK, connect, enable ------------------------------------------
    print("Importing Piper SDK adapter ...")
    from piper_sdk_py_driver.sdk_adapter import PiperSdkAdapter

    adapter = PiperSdkAdapter(
        can_port=parsed.can_port,
        gripper_exist=True,
        enable_timeout=10.0,
    )
    adapter.connect()
    print(f"Connected to {parsed.can_port}")

    print("Enabling arm (blocking, enable_gripper=False) ...")
    ok = adapter.enable(blocking=True, enable_gripper=False)
    if not ok:
        print("ERROR: arm enable failed (timeout or driver not ready)")
        adapter.disconnect()
        node.destroy_node()
        rclpy.shutdown()
        return 1
    print("Arm enabled. Waiting for safe commands from ACT...")

    # --- spin event loop ------------------------------------------------------
    try:
        while not done and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt — stopping session.")
        error_exit = True

    # --- cleanup (keep arm enabled) -------------------------------------------
    adapter._enabled = False
    adapter.disconnect()
    print("Disconnected from CAN (arm left ENABLED)")
    node.destroy_node()
    rclpy.shutdown()

    # --- print summary --------------------------------------------------------
    print(
        "\n=============================================================="
        "\n  REAL-WRITE SESSION SUMMARY"
        "\n=============================================================="
    )
    print(f"  Writes executed: {write_count} / {max_write_steps}")
    print(f"  Command scale:   {command_scale:.1%}")
    print(f"  Freeze gripper:  {freeze_gripper}")
    print(f"  Gripper initial: {gripper_initial:.6f}" if gripper_initial is not None else "  Gripper initial: N/A")
    if error_exit:
        print("  Status:          ERROR / EARLY STOP")
    else:
        print("  Status:          OK")
    if session_start_qpos is not None and step_log:
        final_sn = step_log[-1][6]
        if final_sn is not None:
            cumul = [final_sn[i] - session_start_qpos[i] for i in range(7)]
            print(f"  Session start qpos:    {[f'{v:.5f}' for v in session_start_qpos]}")
            print(f"  Final qpos:            {[f'{v:.5f}' for v in final_sn]}")
            print(f"  Cumulative delta:      {[f'{v:+.5f}' for v in cumul]}")
            print(f"  Cumul J2: {cumul[2]:+.5f} (limit {max_cumul_j2:.4f})  "
                  f"Max arm cumul: {max(abs(cumul[i]) for i in range(6)):.5f} (limit {max_cumul_any:.4f})")
            cumul_triggered = abs(cumul[2]) > max_cumul_j2 or \
                max(abs(cumul[i]) for i in range(6)) > max_cumul_any
            print(f"  Cumulative guard triggered: {cumul_triggered}")

    # --- gripper shadow analysis (Phase 13+) ---------------------------------
    if raw_target_log:
        print(
            "\n  Gripper shadow:"
            "\n  (raw=ACT output before safety_gate, safe=post-gate, actual=arm state)"
        )
        print(f"  {'Step':>5} {'raw':>10} {'safe':>10} {'actual':>10} {'raw-safe':>10}")
        print("  " + "-" * 47)
        raw_grips = []
        for step, raw_g, safe_g, actual_g in raw_target_log:
            diff = raw_g - safe_g if not (math.isnan(raw_g) or math.isnan(safe_g)) else float('nan')
            raw_grips.append(raw_g)
            print(f"  {step:5d} {raw_g:10.5f} {safe_g:10.5f} {actual_g:10.5f} {diff:10.5f}")
        # summary stats
        valid_raw = [g for g in raw_grips if not math.isnan(g)]
        if valid_raw:
            min_raw, max_raw = min(valid_raw), max(valid_raw)
            print(f"\n  Raw gripper range: [{min_raw:.5f}, {max_raw:.5f}]")
            if max_raw - min_raw > 0.001:
                # find first step where raw gripper drops
                initial_g = valid_raw[0]
                close_steps = [i+1 for i, g in enumerate(valid_raw) if abs(g - initial_g) > 0.005]
                if close_steps:
                    print(f"  Gripper delta > 0.005 first seen at step {close_steps[0]}")
                else:
                    print(f"  Gripper stayed within ±0.005 of initial ({initial_g:.5f})")
            else:
                print(f"  Gripper held steady (range < 0.001m) — ACT kept gripper OPEN")

    if not step_log:
        print("\n  No steps logged.")
        return 1 if error_exit else 0

    print(
        "\n  Step-by-step:"
        "\n  (arm joints only — gripper frozen)"
    )
    header = (f"{'Step':>5} {'Joint':>8}  {'current':>10} {'safe_tgt':>10} "
              f"{'delta':>10} {'scaled':>10} {'after_sn':>10} {'act_delta':>10}")
    print(header)
    print("-" * len(header))

    for step_num, cur, safe, delta, scaled, after_sdk, after_sn in step_log:
        for i in range(6):
            name = JOINT_NAMES[i]
            c = cur[i]
            s = safe[i]
            d = delta[i]
            sc = scaled[i]
            a_sn = after_sn[i] if i < len(after_sn) else float('nan')
            actual_d = (a_sn - c) if not (math.isnan(a_sn) or math.isnan(c)) else float('nan')
            marker = ""
            if abs(d) > 1e-9:
                marker = " <<<" if i == 0 else ""
                print(
                    f"{step_num:>5} {name:>8}  {c:10.6f} {s:10.6f} "
                    f"{d:+10.6f} {sc:10.6f} {a_sn:10.6f} "
                    f"{actual_d:+10.6f}{marker}"
                )
        # gripper line
        grip_cur = cur[6]
        grip_safe = safe[6]
        grip_scaled = scaled[6]
        grip_after_sn = after_sn[6] if len(after_sn) > 6 else float('nan')
        grip_delta_sn = (grip_after_sn - gripper_initial) if not (math.isnan(grip_after_sn) or gripper_initial is None) else float('nan')
        print(
            f"{step_num:>5} {'gripper':>8}  {grip_cur:10.6f} {grip_safe:10.6f} "
            f"{grip_safe - grip_cur:+10.6f} {grip_scaled:10.6f} {grip_after_sn:10.6f} "
            f"{grip_delta_sn:+10.6f}  [frozen]"
        )
        print()

    # --- gripper summary ------------------------------------------------------
    if gripper_initial is not None and step_log:
        final_grip_sn = step_log[-1][6][6] if len(step_log[-1][6]) > 6 else float('nan')
        grip_drift = (final_grip_sn - gripper_initial) if not math.isnan(final_grip_sn) else float('nan')
        print(f"  Gripper before: {gripper_initial:.6f}")
        print(f"  Gripper after:  {final_grip_sn:.6f}")
        print(f"  Gripper drift:  {grip_drift:+.6f} {'OK' if abs(grip_drift) < 0.001 else '*** WARNING ***'}")
    else:
        print("  Gripper: N/A")

    print(f"\n  Arm is still ENABLED — disable manually when ready.")
    return 0


# ---------------------------------------------------------------------------
#  main
# ---------------------------------------------------------------------------

def main(args=None):
    parser = argparse.ArgumentParser(
        description="Piper SDK command node — shadow-write + single-step micro-motion"
    )
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--allow-real-write", action="store_true", default=False)
    parser.add_argument("--confirm-real-write", action="store_true", default=False)
    parser.add_argument("--shadow-only", action="store_true", default=True)

    # --- single-step args --------------------------------------------------
    parser.add_argument("--single-step-only", action="store_true", default=False,
                        help="Execute exactly ONE micro-motion and exit")
    parser.add_argument("--can-port", default="can0")
    parser.add_argument("--joint-name", default="joint2")
    parser.add_argument("--delta", type=float, default=0.003)
    parser.add_argument("--max-real-delta", type=float, default=0.005)
    parser.add_argument("--diagnose-enabled-read", action="store_true", default=False,
                        help="Multi-sample read while arm is still enabled (diagnostic)")

    # --- session-trajectory-test args ---------------------------------------
    parser.add_argument("--session-trajectory-test", action="store_true", default=False,
                        help="Execute a short multi-step trajectory while arm stays enabled")
    parser.add_argument("--step-delta", type=float, default=0.001,
                        help="Per-step joint delta in rad (default: 0.001)")
    parser.add_argument("--num-steps", type=int, default=5,
                        help="Number of steps (default: 5)")
    parser.add_argument("--rate", type=float, default=2,
                        help="Step rate in Hz (default: 2, max 2)")
    parser.add_argument("--max-total-delta", type=float, default=0.005,
                        help="Maximum absolute total displacement (default: 0.005)")

    # --- real-write-session args (Phase 10B) ---------------------------------
    parser.add_argument("--real-write-session", action="store_true", default=False,
                        help="ACT real-write session: subscribe /piper/command_joint_safe, "
                             "write scaled targets via SDK, stop after --max-write-steps")
    parser.add_argument("--command-scale", type=float, default=0.2,
                        help="Scale factor applied to safe_target delta (default: 0.2 = 20%%)")
    parser.add_argument("--max-write-steps", type=int, default=5,
                        help="Maximum number of real-write commands (default: 5)")
    parser.add_argument("--freeze-gripper", action="store_true", default=True,
                        help="Keep gripper frozen at current position (default: True)")
    parser.add_argument("--no-freeze-gripper", dest="freeze_gripper", action="store_false",
                        help="Allow gripper to move (use for arm reset)")
    parser.add_argument("--max-cumulative-delta-j2", type=float, default=0.15,
                        help="Max cumulative J2 displacement from session start (default: 0.15)")
    parser.add_argument("--max-cumulative-delta-any-joint", type=float, default=0.18,
                        help="Max cumulative displacement for any arm joint (default: 0.18)")

    parsed = parser.parse_args(args)

    # --- real-write-session path (Phase 10B) ---------------------------------
    if parsed.real_write_session:
        return _run_real_write_session(parsed)

    # --- session-trajectory-test path ---------------------------------------
    if parsed.session_trajectory_test:
        return _run_session_trajectory_test(parsed)

    # --- single-step path --------------------------------------------------
    if parsed.single_step_only:
        return _run_single_step(parsed)

    # --- shadow-only path --------------------------------------------------
    if parsed.allow_real_write and parsed.confirm_real_write:
        print(
            "\n=============================================================="
            "\n  --allow-real-write --confirm-real-write set, but"
            "\n  --single-step-only NOT set."
            "\n  Running in SHADOW-ONLY mode."
            "\n==============================================================\n"
        )

    print(
        "=============================================================="
        "\n  Piper SDK command node — SHADOW-WRITE ONLY"
        "\n  No SDK imports. No CAN connection. No hardware writes."
        "\n  This phase does NOT send commands to the real arm."
        "\n==============================================================\n"
    )

    rclpy.init(args=sys.argv)
    node = PiperSdkCommandNode(parsed.hz)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
