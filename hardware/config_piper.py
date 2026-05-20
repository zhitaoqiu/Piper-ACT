"""PiperRobotConfig — LeRobot-standard configuration for Piper robotic arm."""
# 配置文件 设定某些默认参数
from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from lerobot.robots.config import RobotConfig


@dataclass
class PiperConfig:
    """Configuration for Piper hardware without the RobotConfig ABC mixin."""

    can_port: str = "can0"
    gripper_exist: bool = True
    joint_limit_rad: float = 3.14
    enable_timeout: float = 10.0
    velocity_pct: int = 50
    gripper_effort: int = 1000
    disable_torque_on_disconnect: bool = True

    cameras: dict[str, CameraConfig] = field(default_factory=dict)


@RobotConfig.register_subclass("piper")
@dataclass
class PiperRobotConfig(RobotConfig, PiperConfig):
    pass
