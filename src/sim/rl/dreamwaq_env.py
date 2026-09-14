"""자동 reset 이전 상태와 action 이력을 보존하는 Isaac Lab 환경."""
import torch
from isaaclab.envs import ManagerBasedRLEnv

from src.sim.rl.mdp.gait import GaitPhase
from src.sim.rl.env_cfg import validate_terrain_border
from src.sim.rl.utils.rollout_diagnostics import RolloutDiagnostics

# action 2차 차분에 필요한 과거 action 개수.
ACTION_HISTORY_LENGTH = 2


class DreamWaQEnv(ManagerBasedRLEnv):
    """Isaac Lab step 순서를 유지하며 종료 직전 critic만 보존한다."""

    def __init__(self, cfg, **kwargs):
        """부모 초기화 전후로 필요한 보조 상태를 할당한다."""
        # 최종 설정으로 로봇이 episode 안에 지형 밖에 도달할 수 없는지 확인한다.
        validate_terrain_border(cfg)
        self._capture_terminal = False
        self._terminal_critic = None
        self._gait = None
        super().__init__(cfg, **kwargs)
        critic_dim = self.observation_manager.group_obs_dim["critic"][0]
        self._terminal_critic = torch.zeros(self.num_envs, critic_dim, device=self.device)
        self._terminal_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._previous_previous_action = torch.zeros_like(self.action_manager.action)
        self._action_history_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # 학습 기록용 보행 진단값 수집기를 만든다.
        self._diagnostics = RolloutDiagnostics(self)

    @property
    def gait(self):
        """관측과 보상이 공유할 보행 위상 생성기를 제공한다.

        관측 관리자가 관측 차원을 재는 초기화 단계에서 처음 호출되므로,
        scene이 준비된 시점에 생성한다.
        """
        if self._gait is None:
            self._gait = GaitPhase(self.cfg.gait, self)
        return self._gait

    @property
    def previous_previous_action(self):
        """현재 보상 계산 시 a_(t-2)를 제공한다."""
        return self._previous_previous_action

    @property
    def action_history_valid(self):
        """2차 차분에 필요한 과거 action이 모두 쌓인 환경만 True인 mask를 반환한다."""
        return self._action_history_steps >= ACTION_HISTORY_LENGTH

    def step(self, action):
        """종료 관측과 2차 action 차분용 상태를 갱신한다."""
        # Isaac Lab step 전에 manager 로그를 비우고 물리 진행 동안 유효한 명령을 진단용으로 복사한다.
        self.extras["log"] = {}
        self._diagnostics.begin_step()

        # terminal critic mask를 비우고 a_(t-2)를 보관한 뒤 Isaac Lab step을 실행한다.
        self._terminal_valid.zero_()
        self._previous_previous_action.copy_(self.action_manager.prev_action)
        self._capture_terminal = True
        try:
            obs, reward, terminated, truncated, extras = super().step(action)
        finally:
            self._capture_terminal = False
        # 과거 action이 쌓인 스텝 수를 세고 종료된 환경은 다시 0부터 센다.
        self._action_history_steps += 1
        self._action_history_steps[terminated | truncated] = 0
        extras["terminal_critic"] = self._terminal_critic
        extras["terminal_valid"] = self._terminal_valid

        # 종료되지 않은 환경의 진단값을 계측하고 스텝 평균 진단 지표를 함께 전달한다.
        self._diagnostics.capture(~(terminated | truncated))
        extras["diagnostics"] = self._diagnostics.metrics()

        return obs, reward, terminated, truncated, extras

    def _reset_idx(self, env_ids):
        """manager와 로봇이 reset되기 전에 terminal critic을 복사한다."""
        if self._capture_terminal:
            # 어느 환경도 reset되기 전에 명령 curriculum의 이번 스텝 추적 점수를 누적한다.
            self.command_manager.get_term("base_velocity").capture_step_metrics()

            # reset 직전 critic 관측을 terminal critic으로 보관한다.
            critic = self.observation_manager.compute_group("critic", update_history=False)
            self._terminal_critic[env_ids] = critic[env_ids]
            self._terminal_valid[env_ids] = True

            # reset 직전 상태로 종료 환경의 진단값을 계측한다.
            self._diagnostics.capture(env_ids)
        super()._reset_idx(env_ids)
        # reset된 환경의 위상이 0으로 돌아간 뒤 관측이 계산되도록 캐시를 버린다.
        if self._gait is not None:
            self._gait.invalidate()
        if hasattr(self, "_action_history_steps"):
            self._action_history_steps[env_ids] = 0
            self._previous_previous_action[env_ids] = 0
