"""High-level wrapper around PiperSdkAdapter with safety checks."""

import sys
import time
from typing import List, Optional, Tuple

sys.path.insert(0, "/home/huatec/piper_act_bottle_grasp/piper_sdk_py_driver")
from piper_sdk_py_driver.sdk_adapter import PiperSdkAdapter, JointState, EndPose, ArmStatus


class PiperRobot:
    """Wrapper for Piper robotic arm with built-in safety limits and simpler API."""

    def __init__(
        self,
        can_port: str = "can0",
        gripper_exist: bool = True,
        joint_limit_rad: float = 3.14,
        enable_timeout: float = 10.0,
    ):
        self._adapter = PiperSdkAdapter(
            can_port=can_port,
            gripper_exist=gripper_exist,
            enable_timeout=enable_timeout,
        )
        self.joint_limit = joint_limit_rad
        self.gripper_exist = gripper_exist
        self.can_port = can_port

    # ---- lifecycle ----

    def connect(self) -> None:
        self._adapter.connect()
        time.sleep(0.3)

    def disconnect(self) -> None:
        self._adapter.disable()
        self._adapter.disconnect()

    def enable(self, blocking: bool = True) -> bool:
        return self._adapter.enable(blocking=blocking)

    def disable(self) -> None:
        self._adapter.disable()

    @property
    def is_enabled(self) -> bool:
        return self._adapter.is_enabled

    @property
    def is_ok(self) -> bool:
        return self._adapter.is_ok()

    # ---- state reading ----

    def get_joint_positions(self) -> List[float]:
        """Return [j1..j6, gripper_m]."""
        js = self._adapter.read_joint_state()
        return list(js.position)

    def get_joint_state(self) -> JointState:
        return self._adapter.read_joint_state()

    def get_end_pose(self) -> EndPose:
        return self._adapter.read_end_pose()

    def get_arm_status(self) -> ArmStatus:
        return self._adapter.read_arm_status()

    # ---- command ----

    def set_joint_positions(
        self,
        positions: List[float],
        velocity_pct: int = 30,
        gripper_effort: int = 1000,
    ) -> bool:
        """
        Send joint position command with safety check.
        Returns True if command was within limits and sent.
        """
        # Safety clamp
        for i in range(min(6, len(positions))):
            if abs(positions[i]) > self.joint_limit:
                print(f"  [WARN] Joint {i+1}={positions[i]:.3f} exceeds limit, clipping")
                positions[i] = max(-self.joint_limit, min(self.joint_limit, positions[i]))

        try:
            self._adapter.send_joint_positions(
                positions,
                velocity_percent=velocity_pct,
                gripper_effort=gripper_effort,
            )
            return True
        except Exception as e:
            print(f"  [ERROR] set_joint_positions failed: {e}")
            return False
