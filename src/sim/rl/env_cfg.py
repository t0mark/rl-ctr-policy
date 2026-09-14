# Copyright (c) 2026 DreamWaQ Project
# SPDX-License-Identifier: BSD-3-Clause

"""보행(velocity-tracking) 태스크의 로봇 무관 공용 Isaac Lab ManagerBasedRLEnv 설정.

특정 로봇의 설정이 아닌 공용 설정만 결정
- Scene의 robot 필드는 MISSING
- 로봇별 값은 `configs/{robot}_env_cfg.py`에서 오버라이드한다 (ArticulationCfg, 관절 이름, 보상 가중치 등)

Two-Phase(DreamWaQ 확장) 방식
- actor는 지면 속도·지형 정보를 직접 보지 않고, 보행 위상이 포함된 관측 이력을 입력받는다
- critic만 특권 정보 (실제 속도, 몸통 주변 지형, 좌우 발 주변 지형)를 관측
- 학습 단계는 configure_training_phase()가 지형과 기준 동작 보상으로 구분한다
"""

import copy
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
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise

# Isaac Lab이 기본 제공하는 mdp 항목 함수/커맨드 cfg를 그대로 재사용한다
# (feet_air_time_positive_biped, joint_deviation_l1, UniformVelocityCommandCfg 등 포함).
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from src.sim.rl.mdp import (
    rewards as paper_rewards,
    events as paper_events,
    observations as paper_observations,
)
from src.sim.rl.mdp.gait import GaitCfg
from src.sim.rl.mdp.velocity_command import CurriculumVelocityCommandCfg
from src.sim.rl.terrains import PHASE_INITIAL_TERRAIN_LEVELS, PHASE_TERRAINS_CFGS


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """지형 + 로봇(로봇별 파일에서 채움) + 센서 + 조명."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=PHASE_TERRAINS_CFGS[1],
        max_init_terrain_level=PHASE_INITIAL_TERRAIN_LEVELS[1],
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
    # 좌우 발 국소 지형 스캐너: critic만 사용하며 prim_path는 로봇별 파일에서 발 링크로 오버라이드한다.
    left_foot_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.3, 0.2]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    right_foot_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.3, 0.2]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    # history_length는 제어 주기 한 번의 물리 스텝 수에 맞춰 __post_init__에서 설정한다.
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", track_air_time=True)

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

    # 10초마다 명령을 재샘플링하고 평균 추적 점수에 따라 명령 범위를 넓히는 curriculum 명령이다.
    base_velocity = CurriculumVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class ActionsCfg:
    """액션 사양. joint_names는 로봇별 파일에서 실제 제어 대상 관절로 좁혀서 오버라이드한다."""

    joint_pos = paper_events.DelayedJointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """비대칭 actor-critic을 위해 actor 부분 관측·critic 특권 관측·속도 타깃을 분리하고 이력은 wrapper가 관리한다."""

    @configclass
    class PolicyCurrentCfg(ObsGroup):
        """actor 부분 관측으로, 실제 로봇 센서로 얻을 수 있는 현재 시점의 단일 noisy 표본."""

        gait_phase = ObsTerm(func=paper_observations.gait_phase)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Gnoise(mean=0.0, std=0.2))
        base_roll_pitch = ObsTerm(func=paper_observations.base_roll_pitch, noise=Gnoise(mean=0.0, std=0.06))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Gnoise(mean=0.0, std=0.05))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Gnoise(mean=0.0, std=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            """현재 관측을 연결하고 센서 노이즈를 활성화한다."""
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCurrentCfg):
        """critic 전용 특권 관측으로, actor 관측에 속도·주변 및 발 지형을 추가한 무잡음 값."""

        # 실제 로봇 센서로 직접 얻을 수 없어 critic에만 제공하는 특권 항목이다.
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        height_scan = ObsTerm(func=mdp.height_scan,
                              params={"sensor_cfg": SceneEntityCfg("height_scanner")}, clip=(-1.0, 1.0))
        left_foot_height_scan = ObsTerm(func=paper_observations.foot_height_scan,
                                        params={"sensor_cfg": SceneEntityCfg("left_foot_scanner")}, clip=(-1.0, 1.0))
        right_foot_height_scan = ObsTerm(func=paper_observations.foot_height_scan,
                                         params={"sensor_cfg": SceneEntityCfg("right_foot_scanner")}, clip=(-1.0, 1.0))

        def __post_init__(self):
            """critic에는 센서 노이즈를 적용하지 않는다."""
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class VelocityTargetCfg(ObsGroup):
        """추정기 속도 손실의 정답으로 쓰는 현재 몸체 좌표계 선속도."""

        velocity = ObsTerm(func=mdp.base_lin_vel)

        def __post_init__(self):
            """노이즈 없는 m/s 단위 속도 벡터를 반환한다."""
            self.enable_corruption = False
            self.concatenate_terms = True

    # actor는 policy_current 이력만, critic은 critic 이력만 입력받는다.
    policy_current: PolicyCurrentCfg = PolicyCurrentCfg()
    critic: CriticCfg = CriticCfg()
    velocity_target: VelocityTargetCfg = VelocityTargetCfg()

@configclass
class EventCfg:
    """도메인 랜덤화 이벤트. 기체 특성은 환경마다 한 번 뽑아 고정하고, 외란만 매 제어 주기 갱신한다."""

    # PD 게인 랜덤화는 환경 초기화 시 한 번 적용한다.
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
            "static_friction_range": (0.1, 2.0),
            "dynamic_friction_range": (0.1, 2.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
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
            "com_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.02, 0.02)},
        },
    )
    # 링크 질량 전체를 배율로 흔들어 실제 기체와의 질량 오차를 모사한다.
    randomize_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    base_external_force_torque = EventTerm(
        func=paper_events.randomize_disturbance, mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "force_range": (0.0, 0.0)},
    )
    # action이 출력될 때마다 각 링크에 무작위 힘을 적용한다. 주기는 __post_init__에서 맞춘다.
    disturbance = EventTerm(
        func=paper_events.randomize_disturbance, mode="interval", interval_range_s=(0.02, 0.02),
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "force_range": (-5.0, 5.0), "probability": 1.0},
    )
    motor_strength = EventTerm(func=paper_events.randomize_motor_strength, mode="startup",
                              params={"factor_range": (0.8, 1.2)})
    torque_delay = EventTerm(func=paper_events.randomize_torque_delay, mode="startup",
                            params={"delay_range_s": (0.0, 0.010)})
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
@configclass
class RewardsCfg:
    """Two-Phase Table 4를 기준으로 보상 가중치와 명시적인 자세 오차 벌점을 정의한다."""

    # 명령과 실제 속도를 중력 정렬 yaw 좌표계에서 비교해 몸통 기울기의 영향을 배제한다.
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=2.4,
                                  params={"command_name": "base_velocity", "std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.1,
                                 params={"command_name": "base_velocity", "std": 0.5})
    # 수평 중력 성분은 자세 오차이므로 음의 가중치로 기울어짐을 억제한다.
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    # 관절 가속도와 관절 파워는 DreamWaQ Table I의 수식과 가중치를 사용한다.
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_power = RewTerm(func=paper_rewards.joint_power, weight=-2.0e-5,
                         params={"asset_cfg": SceneEntityCfg("robot")})
    velocity_mismatch = RewTerm(func=paper_rewards.velocity_mismatch, weight=0.5,
                                params={"asset_cfg": SceneEntityCfg("robot")})
    # 기본 자세 추종은 지수 보상이며 기준 동작 추종보다 작은 가중치를 쓴다.
    default_joint_tracking = RewTerm(func=paper_rewards.default_joint_tracking, weight=0.5,
                                    params={"asset_cfg": SceneEntityCfg("robot")})
    body_height = RewTerm(func=paper_rewards.body_height, weight=-1.0,
                         params={"target_height": 0.75, "asset_cfg": SceneEntityCfg("robot"),
                                 "sensor_cfg": SceneEntityCfg("height_scanner")})
    feet_clearance = RewTerm(func=paper_rewards.feet_clearance, weight=-0.01,
                            params={"target_height": 0.08, "sole_offset": 0.0,
                                    "asset_cfg": SceneEntityCfg("robot", body_names=".*FOOT"),
                                    "left_sensor_cfg": SceneEntityCfg("left_foot_scanner"),
                                    "right_sensor_cfg": SceneEntityCfg("right_foot_scanner")})
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    action_smoothness = RewTerm(func=paper_rewards.action_smoothness, weight=-0.01)

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

    training_phase: int = 1
    dwaq_actor_history_length: int = 5
    dwaq_estimator_history_length: int = 5
    dwaq_critic_history_length: int = 3
    gait: GaitCfg = GaitCfg()
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
        # 정책은 100 Hz, PD 제어기는 1000 Hz로 동작한다.
        self.decimation = 10
        self.episode_length_s = 30.0
        self.sim.dt = 0.001
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        for scanner in (self.scene.height_scanner, self.scene.left_foot_scanner,
                        self.scene.right_foot_scanner):
            if scanner is not None:
                scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
            self.scene.contact_forces.history_length = self.decimation

        # 외란은 action이 출력될 때마다 갱신되어야 하므로 제어 주기와 항상 같게 둔다.
        control_period = self.decimation * self.sim.dt
        self.events.disturbance.interval_range_s = (control_period, control_period)

        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


# 실제 이동 속도가 명령 최대 속도를 넘을 수 있는 비율이다.
_SPEED_OVERSHOOT_FACTOR = 1.1
# 스캐너가 붙은 링크(발 등)가 로봇 root에서 수평으로 떨어질 수 있는 최대 거리(m)이다.
_SCANNER_LINK_REACH_M = 1.0


def validate_terrain_border(cfg):
    """한 episode 동안 로봇과 지형 스캐너가 지형 메시 밖에 도달할 수 없는지 검증한다.

    가장 바깥 sub-terrain 중심에서 출발해 최대 명령 속도로 episode 끝까지 직진해도
    스캐너 영역 전체가 테두리 안에 남아야 한다. 조건을 만족하지 않으면 예외를 낸다.
    """
    terrain = cfg.scene.terrain
    generator = terrain.terrain_generator
    if terrain.terrain_type != "generator" or generator is None:
        return

    # 명령 범위에서 가능한 최대 평면 속도로 episode 동안 이동할 수 있는 거리를 구한다.
    ranges = cfg.commands.base_velocity.ranges
    max_speed = math.hypot(max(abs(value) for value in ranges.lin_vel_x),
                           max(abs(value) for value in ranges.lin_vel_y))
    travel_distance = max_speed * cfg.episode_length_s * _SPEED_OVERSHOOT_FACTOR

    # reset 시 sub-terrain 원점에서 벗어날 수 있는 최대 수평 거리를 구한다.
    pose_range = cfg.events.reset_base.params["pose_range"]
    spawn_offset = math.hypot(max(abs(value) for value in pose_range.get("x", (0.0, 0.0))),
                              max(abs(value) for value in pose_range.get("y", (0.0, 0.0))))

    # 스캐너 격자의 최대 반대각선에 링크 도달 거리를 더해 ray 영역의 최대 반경을 구한다.
    scanners = (cfg.scene.height_scanner, cfg.scene.left_foot_scanner, cfg.scene.right_foot_scanner)
    scanner_radius = max(math.hypot(*scanner.pattern_cfg.size) / 2.0
                         for scanner in scanners if scanner is not None)
    scanner_reach = scanner_radius + _SCANNER_LINK_REACH_M

    # 가장 바깥 sub-terrain 중심에서 메시 끝까지의 거리와 필요한 거리를 비교한다.
    required_border = travel_distance + spawn_offset + scanner_reach - min(generator.size) / 2.0
    if generator.border_width < required_border:
        raise ValueError(
            f"지형 테두리 폭 {generator.border_width:.1f} m가 episode 내 최대 도달 거리를 덮지 못합니다. "
            f"필요 폭 {required_border:.1f} m (속도 {max_speed:.2f} m/s, episode {cfg.episode_length_s:.1f} s).")

    
def configure_training_phase(cfg, phase):
    """학습 단계에 맞는 지형과 기준 동작 보상 활성 여부를 환경 설정에 적용한다.

    1단계는 쉬운 지형에서 사인파 기준 동작을 추종하고, 2단계는 어려운 지형에서
    기준 동작 추종 보상만 제거해 자유로운 보행을 학습한다. 위상 관측과 위상 기반
    접촉 보상은 두 단계 모두 유지한다.
    """
    if phase not in PHASE_TERRAINS_CFGS:
        raise ValueError(f"지원하지 않는 학습 단계: {phase}")

    # 단계별 지형 생성기의 복사본을 연결하고 지형 curriculum과 초기 난이도 상한을 설정한다.
    generator = copy.deepcopy(PHASE_TERRAINS_CFGS[phase])
    generator.curriculum = getattr(cfg.curriculum, "terrain_levels", None) is not None
    cfg.scene.terrain.terrain_generator = generator
    cfg.scene.terrain.max_init_terrain_level = PHASE_INITIAL_TERRAIN_LEVELS[phase]

    # 기준 동작 추종 보상 설정을 별도로 보관한다.
    reference = cfg.rewards.joint_position_tracking
    if reference is not None:
        cfg._phase1_reference_reward = copy.deepcopy(reference)
    if not hasattr(cfg, "_phase1_reference_reward"):
        raise ValueError("1단계 기준 동작 보상 설정이 없습니다.")

    # 1단계는 기준 동작 추종 보상을 켜고, 2단계는 None으로 두어 보상 계산에서 제외한다.
    cfg.rewards.joint_position_tracking = (
        copy.deepcopy(cfg._phase1_reference_reward) if phase == 1 else None)
    cfg.training_phase = phase
    return cfg


# 평가 기본 동역학에서 로봇 링크에 고정 적용할 마찰계수이다.
_NOMINAL_FRICTION = 1.0

# 평가 기본 동역학에서 제거하는 랜덤화·외란 이벤트 이름이다.
_RANDOMIZATION_EVENT_NAMES = (
    "add_base_mass", "base_com", "randomize_link_mass", "randomize_actuator_gains",
    "motor_strength", "torque_delay", "base_external_force_torque", "disturbance",
)


def configure_evaluation(cfg, command, flat_terrain=False, observation_noise=False):
    """고정 명령·평가 지형·관측 노이즈·기본 동역학의 평가 조건을 환경 설정에 적용한다.

    configure_training_phase()로 단계 지형을 적용한 뒤 호출한다. command는 [vx, vy, yaw rate]이다.
    """
    # 평지 평가면 지형을 평면으로 바꾸고, 지형 curriculum을 끈다.
    if flat_terrain:
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None
        cfg.scene.terrain.visual_material = None
    cfg.curriculum.terrain_levels = None
    if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False

    # 명령 curriculum과 정지 환경을 끄고 명령을 지정 값으로 고정한다.
    velocity_command = cfg.commands.base_velocity
    velocity_command.curriculum_enabled = False
    velocity_command.rel_standing_envs = 0.0
    velocity_command.ranges.lin_vel_x = (command[0], command[0])
    velocity_command.ranges.lin_vel_y = (command[1], command[1])
    velocity_command.ranges.ang_vel_z = (command[2], command[2])

    # 관측 노이즈 사용 여부를 적용한다.
    cfg.observations.policy_current.enable_corruption = observation_noise

    # 로봇 링크 마찰계수를 고정값으로 설정한다.
    friction = (_NOMINAL_FRICTION, _NOMINAL_FRICTION)
    cfg.events.physics_material.params["static_friction_range"] = friction
    cfg.events.physics_material.params["dynamic_friction_range"] = friction

    # 질량·무게중심·PD 게인·모터 출력·토크 지연 랜덤화와 외란 이벤트를 제거한다.
    for name in _RANDOMIZATION_EVENT_NAMES:
        if getattr(cfg.events, name, None) is None:
            raise AttributeError(f"랜덤화 이벤트 {name}이 환경 설정에 없습니다.")
        setattr(cfg.events, name, None)

    # action 지연을 제거한다.
    cfg.actions.joint_pos.delay_range_s = (0.0, 0.0)
    return cfg
