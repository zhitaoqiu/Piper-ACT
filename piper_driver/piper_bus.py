"""SDK-backed Piper motor bus for Piper driver.

This bus deliberately reuses the locally validated PiperRobot wrapper instead
of copying the reference VA11Hall raw-unit conversion path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hardware.piper_wrapper import PiperRobot

from .schema import MOTOR_POS_KEYS, PIPER_GRIPPER_MAX_M, as_qpos


@dataclass(frozen=True)
class PiperMotorsBusV2Config:
    can_port: str = "can0"
    gripper_exist: bool = True
    joint_limit_rad: float = 3.14
    enable_timeout: float = 10.0
    velocity_pct: int = 50
    gripper_effort: int = 1000
    disable_torque_on_disconnect: bool = False


class PiperMotorsBusV2:
    """Small bus interface exposing rad/meter qpos reads and writes."""

    def __init__(self, config: PiperMotorsBusV2Config):
        self.config = config
        self._robot = PiperRobot(
            can_port=config.can_port,
            gripper_exist=config.gripper_exist,
            joint_limit_rad=config.joint_limit_rad,
            enable_timeout=config.enable_timeout,
            velocity_pct=config.velocity_pct,
            gripper_effort=config.gripper_effort,
            disable_torque_on_disconnect=config.disable_torque_on_disconnect,
        )

    @property
    def is_connected(self) -> bool:
        return self._robot.is_connected

    @property
    def is_enabled(self) -> bool:
        return self._robot.is_enabled

    def connect(self, calibrate: bool = True) -> None:
        self._robot.connect(calibrate=calibrate)

    def disconnect(self) -> None:
        self._robot.disconnect()

    def enable(self, blocking: bool = True) -> bool:
        return self._robot.enable(blocking=blocking)

    def disable(self) -> None:
        self._robot.disable()

    def read_qpos(self) -> np.ndarray:
        return as_qpos(self._robot.get_joint_positions(), label="Piper read qpos")

    def write_qpos(self, qpos, *, velocity_pct: int | None = None) -> np.ndarray:
        target = as_qpos(qpos, label="Piper target qpos").copy()
        target[:6] = np.clip(target[:6], -self.config.joint_limit_rad, self.config.joint_limit_rad)
        target[6] = np.clip(target[6], 0.0, PIPER_GRIPPER_MAX_M)
        sent = self._robot.set_joint_positions(
            target.tolist(),
            velocity_pct=self.config.velocity_pct if velocity_pct is None else velocity_pct,
            gripper_effort=self.config.gripper_effort,
        )
        return as_qpos([sent[key] for key in MOTOR_POS_KEYS], label="Piper sent qpos")
