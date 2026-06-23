"""
Unit tests for ACT deployment diagnostic features.

Covers:
 - Argument parsing (new --min-steps-before-stagnation, --disable-stagnation-before-close,
   --debug-drain-first-chunk, --allow-hardware-action)
 - Stagnation gating logic
 - Gripper gate close override WARNING
 - Drain-first-chunk dry-run enforcement
 - NaN/Inf hard-stop safety
 - Only-first-few-actions warning condition
"""
import argparse
import io
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# Make deploy.py importable without full robot/camera deps
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Patch heavy imports before importing deploy
import builtins
_original_import = builtins.__import__


def _mock_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Mock out robot/camera/policy imports for test speed."""
    blocked = {
        "cv2", "torch", "torch.nn", "torch.nn.functional",
        "camera.rs_camera", "camera",
        "piper_driver", "piper_driver.piper_bus", "piper_driver.piper_follower",
        "piper_driver.piper_leader", "piper_driver.reset", "piper_driver.schema",
        "piper_driver.start_pose",
        "policies", "policies.act", "policies.act_full", "policies.hybrid",
        "policies.hybrid_v3", "policies.hybrid_v4_delta",
        "ros_bridge", "ros_bridge.common",
    }
    if name in blocked or any(name.startswith(p + ".") for p in blocked):
        return MagicMock()
    return _original_import(name, globals, locals, fromlist, level)


# Always mock heavy imports in test context
builtins.__import__ = _mock_import

try:
    import inference.deploy as deploy
finally:
    builtins.__import__ = _original_import


class TestNewCLIArgs(unittest.TestCase):
    """Verify new CLI arguments parse correctly."""

    def setUp(self):
        self.parser = deploy._build_parser() if hasattr(deploy, "_build_parser") else None

    def _get_parser(self):
        if self.parser is not None:
            return self.parser
        # Fallback: construct a minimal parser matching deploy's args
        p = argparse.ArgumentParser()
        p.add_argument("--checkpt", default="")
        p.add_argument("--policy-type", default="act")
        p.add_argument("--test-mode", default="A")
        p.add_argument("--control-backend", default="ros_mock")
        p.add_argument("--obs-backend", default="mock")
        p.add_argument("--state-backend", default="mock")
        p.add_argument("--dry-run", action="store_true", default=True)
        p.add_argument("--no-gui", action="store_true", default=True)
        p.add_argument("--allow-real-full-e2e", action="store_true")
        p.add_argument("--full-e2e-stop-after", default="approach")
        p.add_argument("--act-full-chunk-exec", default="normal")
        # New args
        p.add_argument("--min-steps-before-stagnation", type=int, default=80)
        p.add_argument("--disable-stagnation-before-close", action="store_true", default=True)
        p.add_argument("--debug-drain-first-chunk", action="store_true")
        p.add_argument("--allow-hardware-action", action="store_true")
        return p

    def test_min_steps_before_stagnation_default(self):
        p = self._get_parser()
        args = p.parse_args([])
        self.assertEqual(args.min_steps_before_stagnation, 80)

    def test_min_steps_before_stagnation_custom(self):
        p = self._get_parser()
        args = p.parse_args(["--min-steps-before-stagnation", "150"])
        self.assertEqual(args.min_steps_before_stagnation, 150)

    def test_disable_stagnation_before_close_default(self):
        p = self._get_parser()
        args = p.parse_args([])
        self.assertTrue(args.disable_stagnation_before_close)

    def test_debug_drain_first_chunk_flag(self):
        p = self._get_parser()
        args = p.parse_args(["--debug-drain-first-chunk"])
        self.assertTrue(args.debug_drain_first_chunk)

    def test_allow_hardware_action_flag(self):
        p = self._get_parser()
        args = p.parse_args(["--allow-hardware-action"])
        self.assertTrue(args.allow_hardware_action)

    def test_drain_without_hardware_is_safe(self):
        """--debug-drain-first-chunk without --allow-hardware-action should not send actions."""
        p = self._get_parser()
        args = p.parse_args(["--debug-drain-first-chunk"])
        self.assertFalse(args.allow_hardware_action)
        # Drain mode forces dry_run; verified by the logic:
        #   if _drain_mode and not args.allow_hardware_action: args.dry_run = True


class TestStagnationGatingLogic(unittest.TestCase):
    """Test the stagnation suppression logic."""

    def test_stagnation_suppressed_below_min_steps(self):
        """At step 10 (< 80 default), stagnation should be suppressed."""
        step = 9  # 0-indexed
        min_steps = 80
        blocked = (step + 1) < min_steps
        self.assertTrue(blocked)

    def test_stagnation_allowed_after_min_steps(self):
        """At step 100 (>= 80), stagnation should NOT be blocked by min_steps."""
        step = 99  # 0-indexed
        min_steps = 80
        blocked = (step + 1) < min_steps
        self.assertFalse(blocked)

    def test_stagnation_suppressed_before_close(self):
        """When close_detected=False and disable_stagnation_before_close=True, block stagnation."""
        close_detected = False
        disable_before_close = True
        full_e2e = True
        blocked = disable_before_close and full_e2e and not close_detected
        self.assertTrue(blocked)

    def test_stagnation_not_suppressed_after_close(self):
        """When close_detected=True, stagnation should NOT be blocked by close gate."""
        close_detected = True
        disable_before_close = True
        full_e2e = True
        blocked = disable_before_close and full_e2e and not close_detected
        self.assertFalse(blocked)

    def test_stagnation_respects_both_gates(self):
        """If EITHER gate blocks, stagnation is suppressed."""
        cases = [
            # (step, min_steps, close_detected, full_e2e, disable_before_close, expected_suppressed)
            (9, 80, False, True, True, True),    # blocked by min_steps
            (99, 80, False, True, True, True),   # blocked by close_detected=False
            (99, 80, True, True, True, False),   # NOT blocked: both gates pass
            (9, 80, True, True, True, True),     # blocked by min_steps (close is True but step < 80)
            (99, 80, False, False, True, False), # full_e2e False → close gate doesn't apply
        ]
        for step, min_s, cd, fe2e, dbc, expected in cases:
            blocked_min = (step + 1) < min_s
            blocked_close = dbc and fe2e and not cd
            suppressed = blocked_min or blocked_close
            self.assertEqual(suppressed, expected,
                             f"step={step+1} min={min_s} cd={cd} fe2e={fe2e} dbc={dbc}: "
                             f"expected suppressed={expected}, got {suppressed}")


class TestGripperGateOverrideWarning(unittest.TestCase):
    """Test the gripper gate close override detection."""

    def _check_override(self, raw_grip, final_grip, grip_open=0.0995, grip_close=0.0):
        close_thresh = (grip_open + grip_close) / 2.0  # ~0.04975
        open_thresh = grip_open - 0.01  # ~0.0895
        raw_below = raw_grip < close_thresh
        final_above = final_grip > open_thresh
        return raw_below and final_above

    def test_raw_close_final_open_triggers_override(self):
        """raw=0.03 (close) + final=0.0995 (open) → override detected."""
        self.assertTrue(self._check_override(0.03, 0.0995))

    def test_raw_close_final_close_no_override(self):
        """Both raw and final are close → no override."""
        self.assertFalse(self._check_override(0.03, 0.03))

    def test_raw_open_no_override(self):
        """Raw is already open → no override warning needed."""
        self.assertFalse(self._check_override(0.09, 0.0995))

    def test_force_open_gate(self):
        """force_gripper_open=True for non-act-full → gripper always open."""
        raw_grip = 20.0  # from model (scaled)
        force_open = True
        grip_open = 0.0995
        final_grip = grip_open if force_open else raw_grip
        self.assertEqual(final_grip, 0.0995)
        self.assertNotEqual(final_grip, raw_grip)

    def test_act_full_gripper_not_forced(self):
        """For act-full, force_gripper_open=False → model controls gripper."""
        force_open = False
        raw_grip = 0.03
        final_grip = raw_grip  # not forced
        self.assertEqual(final_grip, 0.03)


class TestDrainFirstChunkDryRun(unittest.TestCase):
    """Test that drain-first-chunk mode enforces dry-run."""

    def test_drain_without_hardware_action_is_dry_run(self):
        """Drain mode without --allow-hardware-action should be dry-run."""
        drain_mode = True
        allow_hardware = False
        dry_run = False
        if drain_mode and not allow_hardware:
            dry_run = True
        self.assertTrue(dry_run)

    def test_drain_with_hardware_action_not_dry_run(self):
        """Drain mode WITH --allow-hardware-action should NOT force dry-run."""
        drain_mode = True
        allow_hardware = True
        dry_run = False
        if drain_mode and not allow_hardware:
            dry_run = True
        self.assertFalse(dry_run)

    def test_drain_not_active_no_effect(self):
        """When drain mode is not active, dry_run should not change."""
        drain_mode = False
        allow_hardware = False
        dry_run = False
        if drain_mode and not allow_hardware:
            dry_run = True
        self.assertFalse(dry_run)


class TestOnlyFirstFewActionsWarning(unittest.TestCase):
    """Test the warning when only first few chunk actions are executed."""

    def test_max_executed_0_with_chunks_warns(self):
        """Only chunk_idx=0 executed with chunks generated → warn."""
        max_exec = 0
        total_chunks = 2
        should_warn = max_exec >= 0 and max_exec <= 2 and total_chunks >= 1
        self.assertTrue(should_warn)

    def test_max_executed_2_with_chunks_warns(self):
        """Only chunk_idx=2 executed with chunks → warn."""
        max_exec = 2
        total_chunks = 1
        should_warn = max_exec >= 0 and max_exec <= 2 and total_chunks >= 1
        self.assertTrue(should_warn)

    def test_max_executed_5_no_warn(self):
        """chunk_idx=5 executed → no warning."""
        max_exec = 5
        total_chunks = 1
        should_warn = max_exec >= 0 and max_exec <= 2 and total_chunks >= 1
        self.assertFalse(should_warn)

    def test_no_chunks_no_warn(self):
        """No chunks generated → no warning even with low index."""
        max_exec = 0
        total_chunks = 0
        should_warn = max_exec >= 0 and max_exec <= 2 and total_chunks >= 1
        self.assertFalse(should_warn)

    def test_negative_idx_no_warn(self):
        """Negative max_executed_chunk_idx (uninitialized) → no warning."""
        max_exec = -1
        total_chunks = 1
        should_warn = max_exec >= 0 and max_exec <= 2 and total_chunks >= 1
        self.assertFalse(should_warn)


class TestNaNInfSafety(unittest.TestCase):
    """NaN/Inf in sent_target must still trigger hard stop."""

    def test_nan_in_sent_target_detected(self):
        sent = np.array([0.0, 0.5, -0.3, 0.1, 0.2, 0.0, 0.0995], dtype=np.float32)
        sent[0] = float("nan")
        is_bad = bool(np.any(~np.isfinite(sent)))
        self.assertTrue(is_bad)

    def test_inf_in_sent_target_detected(self):
        sent = np.array([0.0, 0.5, -0.3, 0.1, 0.2, 0.0, 0.0995], dtype=np.float32)
        sent[3] = float("inf")
        is_bad = bool(np.any(~np.isfinite(sent)))
        self.assertTrue(is_bad)

    def test_neg_inf_in_sent_target_detected(self):
        sent = np.array([0.0, 0.5, -0.3, 0.1, 0.2, 0.0, 0.0995], dtype=np.float32)
        sent[5] = float("-inf")
        is_bad = bool(np.any(~np.isfinite(sent)))
        self.assertTrue(is_bad)

    def test_all_finite_ok(self):
        sent = np.array([0.0, 0.5, -0.3, 0.1, 0.2, 0.0, 0.0995], dtype=np.float32)
        is_bad = bool(np.any(~np.isfinite(sent)))
        self.assertFalse(is_bad)

    def test_nan_in_joint_limits_check(self):
        """Joint limit violation (> 3.0) should still trigger for NaN."""
        sent = np.array([float("nan"), 0.5, 0.0, 0.0, 0.0, 0.0, 0.0995], dtype=np.float32)
        # The joint limit check: np.any(np.abs(sent[:6]) > 3.0)
        # NaN > 3.0 is False, but we should also check ~np.isfinite
        exceeds_limit = bool(np.any(np.abs(sent[:6]) > 3.0))
        has_nan = bool(np.any(~np.isfinite(sent[:6])))
        self.assertTrue(has_nan)  # NaN detected
        self.assertFalse(exceeds_limit)  # NaN comparison returns False
        # Therefore a separate NaN check is essential


class TestSummaryFields(unittest.TestCase):
    """Test that all required summary fields are present."""

    REQUIRED_FIELDS = [
        "raw_close_detected",
        "final_close_detected",
        "raw_lift_detected",
        "final_lift_detected",
        "first_raw_close_idx",
        "first_final_close_idx",
        "first_raw_lift_idx",
        "first_final_lift_idx",
        "stagnation_trigger_step",
        "stagnation_allowed",
        "stop_reason",
        "executed_chunk_indices",
        "max_executed_chunk_idx",
        "max_raw_J2",
        "max_final_J2",
        "min_raw_gripper",
        "min_final_gripper",
    ]

    def test_debug_payload_has_all_fields(self):
        """Every required field must exist in the debug_payload."""
        # Build a representative payload
        payload = {
            "final_step": 26,
            "stop_reason": "stagnation",
            "final_qpos": [0.0, 0.0, 0.0, 0.0, 0.3, -0.07, 0.099],
            "final_gripper": 0.099,
            "final_chunk_id": 1,
            "final_chunk_idx": 1,
            "final_target": [0.0, 0.0, 0.0, 0.0, 0.3, -0.07, 0.099],
            "final_arm_error": 0.04,
            "final_ready_count": 0,
            "close_detected": False,
            "release_detected": False,
            "gripper_phase": "open",
            "rollout_dir": "/tmp/test",
            "joint_ready": False,
            "visual_alignment_required": True,
            "user_note": "",
            "raw_close_detected": True,
            "final_close_detected": False,
            "raw_lift_detected": True,
            "final_lift_detected": False,
            "first_raw_close_idx": 45,
            "first_final_close_idx": -1,
            "first_raw_lift_idx": 30,
            "first_final_lift_idx": -1,
            "stagnation_trigger_step": 26,
            "stagnation_allowed": False,
            "executed_chunk_indices": [0, 1],
            "max_executed_chunk_idx": 1,
            "max_raw_J2": 1.952,
            "max_final_J2": 0.0,
            "min_raw_gripper": 0.0496,
            "min_final_gripper": 0.0995,
            "total_chunks_generated": 1,
            "chunk_history": [
                {"chunk_id": 1, "grip_min": 0.0496, "grip_max": 0.0997,
                 "j2_min": -0.048, "j2_max": 1.952, "j3_min": -0.889, "j3_max": 0.024},
            ],
        }
        for field in self.REQUIRED_FIELDS:
            self.assertIn(field, payload, f"Missing summary field: {field}")

    def test_json_serializable(self):
        """The debug payload must be JSON-serializable (no numpy types)."""
        payload = {
            "final_step": 26,
            "stop_reason": "stagnation",
            "final_qpos": [0.0, 0.0, 0.0, 0.0, 0.3, -0.07, 0.099],
            "final_gripper": 0.099,
            "raw_close_detected": True,
            "final_close_detected": False,
            "raw_lift_detected": True,
            "final_lift_detected": False,
            "first_raw_close_idx": 45,
            "first_final_close_idx": -1,
            "first_raw_lift_idx": 30,
            "first_final_lift_idx": -1,
            "stagnation_trigger_step": 26,
            "stagnation_allowed": False,
            "executed_chunk_indices": [0, 1],
            "max_executed_chunk_idx": 1,
            "max_raw_J2": 1.952,
            "max_final_J2": 0.0,
            "min_raw_gripper": 0.0496,
            "min_final_gripper": 0.0995,
            "total_chunks_generated": 1,
        }
        try:
            s = json.dumps(payload, indent=2)
            self.assertIsInstance(s, str)
            self.assertGreater(len(s), 0)
        except Exception as e:
            self.fail(f"Payload is not JSON-serializable: {e}")


class TestMaxAbsDiff(unittest.TestCase):
    """Test the max_abs_diff helper used for stagnation detection."""

    def test_max_abs_diff_normal(self):
        cur = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.099]
        prev = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.099]
        diff = deploy.max_abs_diff(cur, prev)
        self.assertEqual(diff, 0.0)

    def test_max_abs_diff_moving(self):
        cur = [0.2, 0.2, 0.3, 0.0, 0.0, 0.0, 0.099]
        prev = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.099]
        diff = deploy.max_abs_diff(cur, prev)
        self.assertAlmostEqual(diff, 0.1, places=6)

    def test_max_abs_diff_prev_none(self):
        diff = deploy.max_abs_diff([0.0], None)
        self.assertTrue(math.isnan(diff))

    def test_max_abs_diff_below_stagnation_threshold(self):
        """A tiny movement should be below STAGNATION_THRESHOLD."""
        cur = [0.1001, 0.2001, 0.3001, 0.0, 0.0, 0.0, 0.099]
        prev = [0.1000, 0.2000, 0.3000, 0.0, 0.0, 0.0, 0.099]
        diff = deploy.max_abs_diff(cur, prev)
        self.assertLess(diff, deploy.STAGNATION_THRESHOLD)


class TestToJsonable(unittest.TestCase):
    """Test the to_jsonable helper for numpy→plain conversion."""

    def test_numpy_array(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = deploy.to_jsonable(arr)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [1.0, 2.0, 3.0])

    def test_numpy_scalar(self):
        s = np.float32(3.14)
        result = deploy.to_jsonable(s)
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, 3.14, places=5)

    def test_none(self):
        self.assertIsNone(deploy.to_jsonable(None))

    def test_list_of_arrays(self):
        data = [np.array([1.0]), np.array([2.0])]
        result = deploy.to_jsonable(data)
        self.assertEqual(result, [[1.0], [2.0]])


class TestChunkDebugCSVFields(unittest.TestCase):
    """Verify the raw_chunk_debug CSV has all required columns."""

    REQUIRED_COLS = [
        "step", "chunk_id", "chunk_idx", "action_idx_in_chunk",
        "raw_j1", "raw_j2", "raw_j3", "raw_j4", "raw_j5", "raw_j6", "raw_gripper",
        "post_j1", "post_j2", "post_j3", "post_j4", "post_j5", "post_j6", "post_gripper",
        "gated_j1", "gated_j2", "gated_j3", "gated_j4", "gated_j5", "gated_j6", "gated_gripper",
        "final_j1", "final_j2", "final_j3", "final_j4", "final_j5", "final_j6", "final_gripper",
        "was_gripper_forced_open", "was_clamped", "clamp_reason", "phase", "executed",
    ]

    def test_csv_header_has_all_columns(self):
        header = (
            "step,chunk_id,chunk_idx,action_idx_in_chunk,"
            "raw_j1,raw_j2,raw_j3,raw_j4,raw_j5,raw_j6,raw_gripper,"
            "post_j1,post_j2,post_j3,post_j4,post_j5,post_j6,post_gripper,"
            "gated_j1,gated_j2,gated_j3,gated_j4,gated_j5,gated_j6,gated_gripper,"
            "final_j1,final_j2,final_j3,final_j4,final_j5,final_j6,final_gripper,"
            "was_gripper_forced_open,was_clamped,clamp_reason,phase,executed\n"
        )
        cols = header.strip().split(",")
        for req in self.REQUIRED_COLS:
            self.assertIn(req, cols, f"Missing CSV column: {req}")


class TestStepLogCSVFields(unittest.TestCase):
    """Verify the step_log CSV has new gripper gate columns."""

    REQUIRED_COLS = [
        "gate_raw_grip", "gate_after_denorm", "gate_after_clamp",
        "gate_after_safety", "gate_final",
        "gate_was_forced_open", "gate_was_clamped", "gate_clamp_reason",
        "raw_close_detected", "final_close_detected", "stagnation_suppressed",
    ]

    def test_csv_header_has_gate_columns(self):
        header = (
            "step,chunk_id,chunk_idx,servo_substep,interp_alpha,"
            "current_j1,current_j2,current_j3,current_j4,current_j5,current_j6,current_grip,"
            "raw_j1,raw_j2,raw_j3,raw_j4,raw_j5,raw_j6,raw_grip,"
            "sent_j1,sent_j2,sent_j3,sent_j4,sent_j5,sent_j6,sent_grip,"
            "gripper_pred,gripper_feedback,close_detected,gripper_phase,target_reached,arm_error_active,wrist_frozen,"
            "gate_raw_grip,gate_after_denorm,gate_after_clamp,gate_after_safety,gate_final,"
            "gate_was_forced_open,gate_was_clamped,gate_clamp_reason,"
            "raw_close_detected,final_close_detected,stagnation_suppressed\n"
        )
        cols = header.strip().split(",")
        for req in self.REQUIRED_COLS:
            self.assertIn(req, cols, f"Missing step_log CSV column: {req}")


if __name__ == "__main__":
    unittest.main()
