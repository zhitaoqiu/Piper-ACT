"""LeRobot-style Piper follower for adapter v2."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from functools import cached_property

from lerobot.cameras import CameraConfig
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceNotConnectedError

from .piper_bus import PiperMotorsBusV2, PiperMotorsBusV2Config
from .schema import MOTOR_POS_KEYS, action_to_qpos, qpos_to_action

logger = logging.getLogger(__name__)


@RobotConfig.register_subclass("piper_follower_v2")
@dataclass
class PiperFollowerV2Config(RobotConfig):
    can_port: str = "can0"
    gripper_exist: bool = True
    joint_limit_rad: float = 3.14
    enable_timeout: float = 10.0
    velocity_pct: int = 50
    gripper_effort: int = 1000
    disable_torque_on_disconnect: bool = False
    cameras: dict[str, CameraConfig] = field(default_factory=dict)


class PiperFollowerV2(Robot):
    """Follower robot with Piper-specific bus and standard LeRobot cameras."""

    config_class = PiperFollowerV2Config
    name = "piper_follower_v2"

    def __init__(self, config: PiperFollowerV2Config):
        super().__init__(config)
        self.config = config
        self.bus = PiperMotorsBusV2(
            PiperMotorsBusV2Config(
                can_port=config.can_port,
                gripper_exist=config.gripper_exist,
                joint_limit_rad=config.joint_limit_rad,
                enable_timeout=config.enable_timeout,
                velocity_pct=config.velocity_pct,
                gripper_effort=config.gripper_effort,
                disable_torque_on_disconnect=config.disable_torque_on_disconnect,
            )
        )
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {key: float for key in MOTOR_POS_KEYS}

    @property
    def _cameras_ft(self) -> dict[str, tuple[int, int, int]]:
        return {
            name: (self.config.cameras[name].height, self.config.cameras[name].width, 3)
            for name in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            return
        self.bus.connect(calibrate=calibrate)
        for cam in self.cameras.values():
            cam.connect()
        self.configure()
        logger.info("%s connected on %s.", self, self.config.can_port)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def get_observation(self) -> RobotObservation:
        if not self.bus.is_connected:
            raise DeviceNotConnectedError("Piper follower v2 is not connected.")
        obs: RobotObservation = qpos_to_action(self.bus.read_qpos())
        for cam_name, cam in self.cameras.items():
            start = time.perf_counter()
            obs[cam_name] = cam.async_read()
            logger.debug("%s read %s in %.1fms", self, cam_name, (time.perf_counter() - start) * 1e3)
        return obs

    def send_action(self, action: RobotAction) -> RobotAction:
        if not self.bus.is_connected:
            raise DeviceNotConnectedError("Piper follower v2 is not connected.")
        sent = self.bus.write_qpos(action_to_qpos(action))
        return qpos_to_action(sent)

    def disconnect(self) -> None:
        for cam in self.cameras.values():
            if cam.is_connected:
                cam.disconnect()
        self.bus.disconnect()
        logger.info("%s disconnected.", self)
