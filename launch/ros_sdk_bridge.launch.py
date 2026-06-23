"""
ROS 2 launch file — full Piper ROS <-> SDK bridge.

Starts all bridge nodes needed to run a real-robot ACT demo.
The ACT inference (deploy.py) runs separately in the ACT conda env.

Data flow:
  deploy.py --UDP:50051--> udp_to_ros_action_node --> /policy/target_joint_raw
                                                             │
  Piper arm <--SDK read-- piper_sdk_state_node --> /piper/joint_states
       │                                                   │
       │              ┌─────────────── safety_gate_node ───┘
       │              ▼
       │       /piper/command_joint_safe
       │              │
       └──SDK write-- piper_sdk_command_node (--real-write-session)

  ros_state_udp_publisher_node --> /piper/joint_states --UDP:50052--> deploy.py

Usage:
  1. Set up the ROS 2 environment FIRST (system Python, NOT the ACT conda env):
     conda deactivate
     source /opt/ros/humble/setup.bash
     source /home/huatec/piper_py_ws/install/setup.bash

  2. Launch the bridge (all safety gates default OFF):
     python3 launch/ros_sdk_bridge.launch.py allow_real_read:=true allow_real_write:=true

  3. In another terminal, run the ACT inference:
     conda activate lerobot_q
     python3 inference/deploy.py \
       --control-backend ros_mock \
       --state-backend real_ros \
       --test-mode full-e2e \
       ...
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node as RosNode

PROJECT_ROOT = "/home/huatec/piper_act_bottle_grasp"


def generate_launch_description():
    # ── Safety gates ────────────────────────────────────────────────────
    allow_real_read = LaunchConfiguration("allow_real_read", default="false")
    allow_real_write = LaunchConfiguration("allow_real_write", default="false")
    can_port = LaunchConfiguration("can_port", default="can0")
    state_hz = LaunchConfiguration("state_hz", default="30")
    command_hz = LaunchConfiguration("command_hz", default="2")
    command_scale = LaunchConfiguration("command_scale", default="0.2")
    max_write_steps = LaunchConfiguration("max_write_steps", default="200")
    max_cumul_j2 = LaunchConfiguration("max_cumul_j2", default="0.15")
    max_cumul_any = LaunchConfiguration("max_cumul_any", default="0.18")

    ld = LaunchDescription([
        DeclareLaunchArgument("allow_real_read", default_value="false",
                              description="Connect to real Piper arm via CAN/SDK to READ joint states"),
        DeclareLaunchArgument("allow_real_write", default_value="false",
                              description="Write commands to real Piper arm via SDK (requires allow_real_read:=true)"),
        DeclareLaunchArgument("can_port", default_value="can0",
                              description="CAN port name"),
        DeclareLaunchArgument("state_hz", default_value="30",
                              description="Joint state publish rate (Hz)"),
        DeclareLaunchArgument("command_hz", default_value="2",
                              description="Command write rate (Hz, max 2 for 10F)"),
        DeclareLaunchArgument("command_scale", default_value="0.2",
                              description="Scale factor applied to safe_target delta (0.2 = 20%)"),
        DeclareLaunchArgument("max_write_steps", default_value="200",
                              description="Max number of real-write commands per session"),
        DeclareLaunchArgument("max_cumul_j2", default_value="0.15",
                              description="Max cumulative J2 displacement from session start (rad)"),
        DeclareLaunchArgument("max_cumul_any", default_value="0.18",
                              description="Max cumulative displacement for any arm joint (rad)"),
    ])

    # ── Node 1: SDK → /piper/joint_states ───────────────────────────────
    node_state = ExecuteProcess(
        cmd=[
            "python3",
            f"{PROJECT_ROOT}/ros_bridge/piper_sdk_state_node.py",
            "--can-port", can_port,
            "--hz", state_hz,
            "--allow-real-read",
        ],
        name="piper_sdk_state_node",
        output="screen",
        condition=IfCondition(allow_real_read),
    )

    # ── Node 2: /piper/joint_states → UDP :50052 (for deploy.py) ────────
    node_state_udp = ExecuteProcess(
        cmd=[
            "python3",
            f"{PROJECT_ROOT}/ros_bridge/ros_state_udp_publisher_node.py",
        ],
        name="ros_state_udp_publisher",
        output="screen",
        condition=IfCondition(allow_real_read),
    )

    # ── Node 3: UDP :50051 → /policy/target_joint_raw ───────────────────
    node_action_in = ExecuteProcess(
        cmd=[
            "python3",
            f"{PROJECT_ROOT}/ros_bridge/udp_to_ros_action_node.py",
        ],
        name="udp_to_ros_action_node",
        output="screen",
    )

    # ── Node 4: /policy/target_joint_raw + /piper/joint_states
    #            → /piper/command_joint_safe ─────────────────────────────
    node_safety = ExecuteProcess(
        cmd=[
            "python3",
            f"{PROJECT_ROOT}/ros_bridge/safety_gate_node.py",
        ],
        name="piper_safety_gate_node",
        output="screen",
    )

    # ── Node 5: /piper/command_joint_safe → SDK write ───────────────────
    # Only starts when allow_real_write is true
    node_command = ExecuteProcess(
        cmd=[
            "python3",
            f"{PROJECT_ROOT}/ros_bridge/piper_sdk_command_node.py",
            "--real-write-session",
            "--can-port", can_port,
            "--rate", command_hz,
            "--command-scale", command_scale,
            "--max-write-steps", max_write_steps,
            "--max-cumulative-delta-j2", max_cumul_j2,
            "--max-cumulative-delta-any-joint", max_cumul_any,
            "--freeze-gripper",
            "--allow-real-write",
            "--confirm-real-write",
        ],
        name="piper_sdk_command_node",
        output="screen",
        condition=IfCondition(allow_real_write),
    )

    # ── Warning when running without real hardware ──────────────────────
    warn_mock = LogInfo(
        "╔══════════════════════════════════════════════════════════════╗\n"
        "║  ROS SDK Bridge — MOCK / DRY-RUN MODE                       ║\n"
        "║  No CAN connection. No hardware writes.                      ║\n"
        "║  To connect real arm, add:                                   ║\n"
        "║    allow_real_read:=true allow_real_write:=true              ║\n"
        "╚══════════════════════════════════════════════════════════════╝"
    )

    # ── Startup summary when real hardware is enabled ───────────────────
    info_real = LogInfo(
        "╔══════════════════════════════════════════════════════════════╗\n"
        "║  ROS SDK Bridge — REAL HARDWARE MODE                        ║\n"
        "║  SDK state node:     CONNECTING TO REAL ARM (read-only)     ║\n"
        "║  SDK command node:   WILL WRITE TO REAL ARM                 ║\n"
        "║  Max write steps:    see --max-write-steps                  ║\n"
        "║  Command scale:      see --command-scale                    ║\n"
        "║  Gripper:            FROZEN                                 ║\n"
        "╚══════════════════════════════════════════════════════════════╝"
    )

    # Show warnings first, then start nodes
    ld.add_action(warn_mock)
    ld.add_action(node_action_in)  # UDP listener — always needed
    ld.add_action(node_safety)     # safety gate — always needed
    ld.add_action(node_state)
    ld.add_action(node_state_udp)
    ld.add_action(node_command)

    return ld
