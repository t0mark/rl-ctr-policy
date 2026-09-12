# Copyright (c) 2026 DreamWaQ Project
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree G1 보행 태스크 설정. src.sim.rl.env_cfg의 로봇 무관 공용 클래스를 상속해서
G1(다리+허리 15 DOF만 RL 제어, 팔은 기본자세 PD 고정)에 맞는 값만 오버라이드한다.

공용 베이스는 DreamWaQ 논문(사족보행 A1)의 값을 그대로 유지하고, 이족보행에서
의미가 달라지는 항목만 Isaac Lab 공식 G1 예제
(reference/isaac_lab/.../velocity/config/g1/rough_env_cfg.py)의 값으로 좁힌다.
각 오버라이드의 계측 근거는 reference/dwaq_g1_test/dreamwaq_reward_tuning.md에 있다.
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

# 관절 가속도 페널티 대상. 공식 G1 예제와 동일하게 다리의 큰 관절만 대상으로 한다.
G1_ACCELERATION_JOINT_NAMES = [".*_hip_.*_joint", ".*_knee_joint"]


@configclass
class G1Rewards(RewardsCfg):
    """G1(이족보행)에 맞게 추가한 보상 항목.

    수치 출처: Isaac Lab 공식 G1 예제의 G1Rewards.
    팔은 RL 제어 대상이 아니라 기본자세로 PD 고정되므로, 공식 예제의 팔 편차 항목은 두지 않는다.
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
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="waist_.*_joint")},
    )


@configclass
class G1EnvCfg(VelocityEnvCfg):
    """G1 관절·링크 매핑과 이족보행에 맞춘 보상 가중치를 연결한다."""

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
        # power는 RL 제어 관절 전체를, 가속도는 다리의 큰 관절만 대상으로 합산한다.
        for reward in (self.rewards.joint_power, self.rewards.power_distribution):
            reward.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=G1_ACTUATED_JOINT_NAMES)
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=G1_ACCELERATION_JOINT_NAMES)
        # 이족보행에서 의미가 달라지는 가중치를 공식 G1 예제 값으로 좁힌다.
        # 상하 진동은 이족보행의 정상 동작이므로 벌하지 않는다.
        self.rewards.lin_vel_z_l2.weight = 0.0
        # 종료의 100%가 자세 이탈이므로 몸통 수평 유지 신호를 강화한다.
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"
        # 휴머노이드는 사족보행용 리셋 랜덤화를 견디지 못하므로 공식 G1 예제 값을 쓴다.
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params["velocity_range"] = {
            axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")}
