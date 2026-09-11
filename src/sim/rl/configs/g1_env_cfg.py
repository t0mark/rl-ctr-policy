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
    """G1 전용 보행 환경 설정."""

    rewards: G1Rewards = G1Rewards()

    def __post_init__(self):
        super().__post_init__()

        # 로봇: 이번 프로젝트의 G1(손가락 없음, 29 DOF) ArticulationCfg
        self.scene.robot = UNITREE_G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # 높이 스캐너: 다리에 덜 가려지는 torso_link에 부착 (Isaac Lab 공식 G1 예제와 동일)
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

        # 액션: 15개 관절(다리+허리)만 RL이 직접 제어. 나머지(팔)는 기본자세로 PD 고정된다.
        self.actions.joint_pos.joint_names = G1_ACTUATED_JOINT_NAMES

        # 도메인 랜덤화: 이족보행 균형 학습 안정성을 위해 초기 단계엔 밀치기/질량변화를 끈다.
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None

        # 보상: 사족보행 기본값 중 G1에 안 맞는 것만 끄거나 조정
        self.rewards.track_ang_vel_z_exp.weight = 2.0  # 기본값 0.5 -> G1 공식 예제 값 2.0
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )
        self.rewards.dof_torques_l2.weight = -1.5e-7
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        )

        # 명령: 전진 위주(뒤로 걷기/옆으로 걷기 없음)
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # 종료: torso_link 접촉 시 종료 (roll/pitch 각도 종료는 아직 미적용 — 5단계 결정 사항이라
        # 여기 이식은 별도 mdp 함수가 필요해서 다음 단계에서 추가한다)
        self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"
