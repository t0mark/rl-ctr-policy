# Copyright (c) 2026 DreamWaQ Project
# SPDX-License-Identifier: BSD-3-Clause

"""보행(velocity-tracking) 태스크의 로봇 무관 공용 Isaac Lab ManagerBasedRLEnv 설정.

특정 로봇의 설정이 아닌 공용 설정만 결정
- Scene의 robot 필드는 MISSING
- 로봇별 값은 `configs/{robot}_env_cfg.py`에서 오버라이드한다 (ArticulationCfg, 관절 이름, 보상 가중치 등)

DreamWaQ 방식
- actor는 지면 속도·지형 정보를 직접 보지 않고(관측 이력을 쌓아 컨텍스트 인코더로 암묵적으로 추정)
- critic만 특권 정보 (실제 속도, 지형 높이)를 관측
"""

import math
from dataclasses import MISSING

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import DomeLightCfg, MdlFileCfg, RigidBodyMaterialCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

# Isaac Lab이 기본 제공하는 mdp 항목 함수/커맨드 cfg를 그대로 재사용한다
# (feet_air_time_positive_biped, joint_deviation_l1, UniformVelocityCommandCfg 등 포함).
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

##
# Scene definition (로봇 무관 공용)
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """지형 + 로봇(로봇별 파일에서 채움) + 센서 + 조명."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=MdlFileCfg(
            mdl_path=(
                f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
                "TilesMarbleSpiderWhiteBrickBondHoned.mdl"
            ),
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # 로봇: 로봇별 파일(configs/g1_env_cfg.py 등)의 __post_init__에서 채운다
    robot: ArticulationCfg = MISSING

    # 높이 스캐너: prim_path는 로봇마다 다른 링크 이름을 가리켜야 하므로 로봇별 파일에서 오버라이드
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings (로봇 무관 공용)
##


@configclass
class CommandsCfg:
    """속도 명령 사양."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class ActionsCfg:
    """액션 사양. joint_names는 로봇별 파일에서 실제 제어 대상 관절로 좁혀서 오버라이드한다."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class ObservationsCfg:
    """policy_current(현재 스텝 관측)/policy(그 이력)/critic 세 그룹으로 나눈 관측 사양.

    dwaq(DreamWaQ PPO)는 "현재 스텝 고유수용감각(obs)"과 "flatten된 관측 이력(obs_hist)"을
    별도 텐서 두 개로 요구한다. Isaac Lab의 그룹 history_length 기능은 term별로 각자의
    이력을 flatten한 뒤 term끼리 이어붙이는 방식이라(시간 순서로 안 묶임), 이 그룹(policy)의
    출력에서 "마지막 스텝"만 잘라내는 방법으로는 현재 스텝 전체를 복원할 수 없다.
    그래서 이력이 없는 policy_current 그룹을 별도로 둬서 "현재 스텝 obs"를 직접 얻는다.
    """

    @configclass
    class PolicyCurrentCfg(ObsGroup):
        """actor가 매 스텝 그대로 쓰는 현재 시점 고유수용감각. 노이즈 포함, 이력 없음.

        아래 PolicyCfg(이력 그룹)와 반드시 동일한 term 구성·순서를 유지해야 한다 — dwaq PPO의
        actor는 이 그룹의 출력을 그대로 쓰고, PolicyCfg는 이 관측들의 시간 이력을 컨텍스트
        인코더(CENet) 입력으로 제공한다.
        """

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PolicyCfg(ObsGroup):
        """PolicyCurrentCfg와 동일한 관측의 이력(history). 컨텍스트 인코더(CENet) 입력 전용.

        base_lin_vel(속도)과 height_scan(지형)은 일부러 안 넣는다 — DreamWaQ 설계상 actor는
        이 정보를 직접 보지 않고, 이 그룹의 이력(history)을 컨텍스트 인코더에 넣어 암묵적으로
        추정한다("Implicit Terrain Imagination").
        """

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            # DreamWaQ 컨텍스트 인코더 입력용 이력 길이. 로봇별 파일에서 관절 수가 정해지면
            # cenet_in_dim = history_length * (이 그룹의 1-step 차원)으로 계산한다.
            self.history_length = 5
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObsGroup):
        """critic이 보는 관측: PolicyCurrentCfg와 같은 항목 + 특권 정보. 노이즈/이력 없음.

        dwaq PPO는 이 그룹의 [num_obs : num_obs+3] 구간을 base_lin_vel(속도 정답)로 그대로
        슬라이싱한다(prev_critic_obs_batch[:, proprio_obs_dim:proprio_obs_dim+3]). 그래서
        term 순서가 반드시 [PolicyCurrentCfg와 동일한 순서의 proprioception 블록] ->
        [base_lin_vel] -> [height_scan] 이어야 한다. 순서를 바꾸면 안 된다.
        """

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy_current: PolicyCurrentCfg = PolicyCurrentCfg()
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """도메인 랜덤화 이벤트."""

    # PD 게인(stiffness/damping) 랜덤화. implicit actuator는 CPU 텐서를 쓰므로
    # 매 스텝이 아닌 환경 초기화 시점(mode="startup")에만 적용한다.
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)},
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    """보상 항목. 로봇별 파일에서 가중치를 오버라이드하거나 항목을 끈다(weight=0/None)."""

    # 몸통이 기울어도(험지 경사, 보행 중 흔들림) 그 기울임에 오염되지 않도록, body-frame이
    # 아니라 중력정렬(yaw-frame)/world-frame 기준으로 속도를 측정하는 함수를 쓴다. 기울임이
    # 0이면 body-frame 버전과 완전히 같은 값을 내는 상위호환이라 4족보행/휴머노이드 공용으로 안전하다.
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.125,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)


@configclass
class TerminationsCfg:
    """종료 조건. body_names/limit_angle은 로봇마다 다른 값으로 오버라이드한다."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    # 자세(roll/pitch) 종료: projected gravity와 world z축 사이 각도가 limit_angle(rad)을 넘으면 종료.
    # 접촉 기반 종료(base_contact)와 달리 넘어지는 중이라도 몸통이 아직 안 닿았으면 여기서 먼저 걸린다.
    base_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.7})


@configclass
class CurriculumCfg:
    """지형 난이도 커리큘럼."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


##
# Environment configuration (로봇 무관 공용 베이스)
##


@configclass
class VelocityEnvCfg(ManagerBasedRLEnvCfg):
    """보행(velocity-tracking) 태스크의 공용 베이스. 로봇별 파일이 이 클래스를 상속한다."""

    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """일반 시뮬레이션 설정과 센서 업데이트 주기를 정리한다."""
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
