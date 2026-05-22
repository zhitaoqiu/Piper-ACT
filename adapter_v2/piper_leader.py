"""Optional LeRobot-style Piper leader for adapter v2."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from lerobot.processor import RobotAction
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.errors import DeviceNotConnectedError

from .piper_bus import PiperMotorsBusV2, PiperMotorsBusV2Config
from .schema import MOTOR_POS_KEYS, qpos_to_action

logger = logging.getLogger(__name__)


@TeleoperatorConfig.register_subclass("piper_leader_v2")
@dataclass
class PiperLeaderV2Config(TeleoperatorConfig):
    can_port: str = ""
    gripper_exist: bool = True
    joint_limit_rad: float = 3.14
    enable_timeout: float = 10.0
    disable_torque_on_disconnect: bool = False


class PiperLeaderV2(Teleoperator):
    """Read a second Piper arm as a leader action source when available."""

    config_class = PiperLeaderV2Config
    name = "piper_leader_v2"

    def __init__(self, config: PiperLeaderV2Config):
        super().__init__(config)
        self.config = config
        self.bus = PiperMotorsBusV2(
            PiperMotorsBusV2Config(
                can_port=config.can_port,
                gripper_exist=config.gripper_exist,
                joint_limit_rad=config.joint_limit_rad,
                enable_timeout=config.enable_timeout,
                disable_torque_on_disconnect=config.disable_torque_on_disconnect,
            )
        )

    @property
    def action_features(self) -> dict[str, type]:
        return {key: float for key in MOTOR_POS_KEYS}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def connect(self, calibrate: bool = True) -> None:
        if not self.config.can_port:
            raise ValueError("PiperLeaderV2 requires an explicit --teleop.can_port after CAN validation.")
        self.bus.connect(calibrate=calibrate)
        self.configure()
        logger.info("%s connected on %s.", self, self.config.can_port)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def get_action(self) -> RobotAction:
        if not self.bus.is_connected:
            raise DeviceNotConnectedError("Piper leader v2 is not connected.")
        return qpos_to_action(self.bus.read_qpos())

    def send_feedback(self, feedback: dict) -> None:
        return None

    def disconnect(self) -> None:
        self.bus.disconnect()
        logger.info("%s disconnected.", self)
