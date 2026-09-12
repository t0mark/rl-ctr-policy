"""자동 reset 이전 상태와 action 이력을 보존하는 Isaac Lab 환경."""
import torch
from isaaclab.envs import ManagerBasedRLEnv

from src.sim.rl.mdp.gait import GaitPhase


class DreamWaQEnv(ManagerBasedRLEnv):
    """Isaac Lab step 순서를 유지하며 종료 직전 critic만 보존한다."""

    def __init__(self, cfg, **kwargs):
        """부모 초기화 전후로 필요한 보조 상태를 할당한다."""
        self._capture_terminal = False
        self._terminal_critic = None
        self._gait = None
        super().__init__(cfg, **kwargs)
        critic_dim = self.observation_manager.group_obs_dim["critic"][0]
        self._terminal_critic = torch.zeros(self.num_envs, critic_dim, device=self.device)
        self._terminal_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._previous_previous_action = torch.zeros_like(self.action_manager.action)
        self._action_history_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

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
        """reset 후 첫 action에서 2차 차분을 제외하는 mask를 반환한다."""
        return self._action_history_valid

    def step(self, action):
        """종료 관측과 2차 action 차분용 상태를 갱신한다."""
        self._terminal_valid.zero_()
        self._previous_previous_action.copy_(self.action_manager.prev_action)
        self._capture_terminal = True
        try:
            obs, reward, terminated, truncated, extras = super().step(action)
        finally:
            self._capture_terminal = False
        self._action_history_valid.copy_(~(terminated | truncated))
        extras["terminal_critic"] = self._terminal_critic
        extras["terminal_valid"] = self._terminal_valid
        return obs, reward, terminated, truncated, extras

    def _reset_idx(self, env_ids):
        """manager와 로봇이 reset되기 전에 terminal critic을 복사한다."""
        if self._capture_terminal:
            critic = self.observation_manager.compute_group("critic", update_history=False)
            self._terminal_critic[env_ids] = critic[env_ids]
            self._terminal_valid[env_ids] = True
        super()._reset_idx(env_ids)
        # reset된 환경의 위상이 0으로 돌아간 뒤 관측이 계산되도록 캐시를 버린다.
        if self._gait is not None:
            self._gait.invalidate()
        if hasattr(self, "_action_history_valid"):
            self._action_history_valid[env_ids] = False
            self._previous_previous_action[env_ids] = 0
