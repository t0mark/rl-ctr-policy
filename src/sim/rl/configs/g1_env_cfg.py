# Copyright (c) 2026 DreamWaQ Project
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree G1 보행 태스크 설정. src.sim.rl.env_cfg의 로봇 무관 공용 클래스를 상속해서
G1(다리+허리 15 DOF만 RL 제어, 팔은 기본자세 PD 고정)에 맞는 값만 오버라이드한다.
"""

import os
import sys

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

# env_cfg.py와 동일한 mdp 모듈(isaaclab 기본 항목 + biped 전용 항목을 함께 재노출함)을 쓴다.
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from src.sim.rl.env_cfg import RewardsCfg, VelocityEnvCfg
from src.sim.rl.paths import ROOT_DIR
from src.sim.rl.mdp.actuators import StrengthPDActuatorCfg

# data/robot/assets/humanoid/unitree_g1/unitree_g1_cfg.py를 찾기 위한 경로 추가.
# (paths.py에서 이미 검증된 ROOT_DIR을 재사용한다 — 이 파일에서 dirname을 직접 다시 세면
# 폴더 깊이가 바뀔 때마다 개수를 세는 실수가 반복되기 쉽다.)
_G1_ASSET_DIR = os.path.join(ROOT_DIR, "data", "robot", "assets", "humanoid", "unitree_g1")
sys.path.insert(0, _G1_ASSET_DIR)
from unitree_g1_cfg import UNITREE_G1_CFG  # noqa: E402

# G1의 RL 제어 대상 관절(다리 12 + 허리 3 = 15). 나머지(팔)는 기본자세로 PD 고정된다.
G1_ACTUATED_JOINT_NAMES = [".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint", "waist_.*_joint"]


@configclass
class G1Rewards(RewardsCfg):
    """G1(이족보행)에 맞게 오버라이드한 보상 항목.

    수치 출처: reference/isaac_lab/source/.../locomotion/velocity/config/g1/rough_env_cfg.py
    (Isaac Lab 공식 G1 예제)의 G1Rewards를 그대로 옮긴 것이다.
    """

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                ],
            )
        },
    )
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="waist_.*_joint")},
    )


@configclass
class G1EnvCfg(VelocityEnvCfg):
    """G1 관절·링크 매핑과 논문 보상 초기값을 연결한다."""

    rewards: G1Rewards = G1Rewards()

    def __post_init__(self):
        """자산 gain을 explicit PD에 연결하고 MDP 대상을 G1에 맞춘다."""
        super().__post_init__()
        self.scene.robot = UNITREE_G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # 토크 배율을 직접 적용하는 explicit PD로 모든 관절을 제어한다.
        self.scene.robot.actuators = {
            name: StrengthPDActuatorCfg(**{
                key: value for key, value in vars(cfg).items()
                if not key.startswith("_") and key != "class_type"
            }) for name, cfg in self.scene.robot.actuators.items()
        }
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"
        self.actions.joint_pos.joint_names = G1_ACTUATED_JOINT_NAMES
        for event in (self.events.add_base_mass, self.events.base_com,
                      self.events.base_external_force_torque, self.events.disturbance):
            event.params["asset_cfg"].body_names = ["torso_link"]
        self.rewards.body_height.params["target_height"] = UNITREE_G1_CFG.init_state.pos[2]
        self.rewards.feet_clearance.params["asset_cfg"].body_names = ".*_ankle_roll_link"
        # power와 가속도는 RL 제어 관절을 대상으로 합산한다.
        for reward in (self.rewards.joint_power, self.rewards.power_distribution, self.rewards.dof_acc_l2):
            reward.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=G1_ACTUATED_JOINT_NAMES)
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"
