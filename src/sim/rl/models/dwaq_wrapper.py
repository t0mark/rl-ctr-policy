"""Isaac Lab 관측을 시점이 명시된 DreamWaQ 전이로 변환한다."""
import copy
import torch

class DwaqVecEnvWrapper:
    """현재 관측을 한 번만 읽어 같은 표본으로 이력을 구성한다."""

    def __init__(self, env):
        """환경을 초기화하고 실제 관측 차원과 배열 순서를 기록한다."""
        self._env = env
        self.device = env.device
        self.step_dt = env.step_dt
        self.num_envs = env.num_envs
        self.num_actions = env.action_manager.total_action_dim
        self.max_episode_length = env.max_episode_length
        self.training_phase = env.cfg.training_phase
        self.actor_history_length = env.cfg.dwaq_actor_history_length
        self.estimator_history_length = env.cfg.dwaq_estimator_history_length
        self.critic_history_length = env.cfg.dwaq_critic_history_length
        obs_dict, _ = env.reset()
        self.num_obs = obs_dict["policy_current"].shape[-1]
        self.num_privileged_obs = obs_dict["critic"].shape[-1] * self.critic_history_length
        # critic용 특권 관측 이력과 actor용 부분 관측 이력을 따로 쌓는다.
        self._critic_history = obs_dict["critic"].unsqueeze(1).repeat(1, self.critic_history_length, 1)
        self._history = obs_dict["policy_current"].unsqueeze(1).repeat(1, self.actor_history_length, 1)
        self._current = self._pack(obs_dict)
        action_term = env.action_manager.get_term("joint_pos")
        self._spec = {
            "obs_dim": self.num_obs, "critic_dim": self.num_privileged_obs,
            "actions": self.num_actions,
            "actor_history_length": self.actor_history_length,
            "estimator_history_length": self.estimator_history_length,
            "critic_history_length": self.critic_history_length,
            "action_joint_names": list(action_term.joint_names),
            "step_dt": self.step_dt,
        }

    @property
    def specification(self):
        """실행 기록용 관측·행동 규격을 반환한다."""
        return copy.deepcopy(self._spec)

    def curriculum_state(self):
        """체크포인트에 저장할 명령 curriculum의 현재 범위를 반환한다."""
        manager = getattr(self._env, "command_manager", None)
        if manager is None:
            return None
        return manager.get_term("base_velocity").curriculum_state()

    def curriculum_metrics(self):
        """명령 curriculum의 평균 추적 점수와 현재 범위를 기록용으로 반환한다."""
        manager = getattr(self._env, "command_manager", None)
        if manager is None:
            return {}
        return manager.get_term("base_velocity").curriculum_metrics()

    def load_curriculum_state(self, state):
        """명령 curriculum 범위를 복원하고 환경을 reset해 새 episode를 시작한다."""
        if state is not None:
            self._env.command_manager.get_term("base_velocity").load_curriculum_state(state)
        self.reset()

    @property
    def episode_length_buf(self):
        """환경의 episode 길이를 노출한다."""
        return self._env.episode_length_buf

    def _pack(self, obs_dict):
        """현재 관측·관측 이력·특권 관측 이력·속도 정답을 한 묶음으로 반환한다."""
        # actor용 부분 관측, critic용 특권 관측, 추정기용 속도 정답을 서로 다른 키로 분리해 전달한다.
        return {"obs": obs_dict["policy_current"], "critic": self._critic_history.flatten(1).clone(),
                "velocity": obs_dict["velocity_target"],
                "history": self._history.flatten(1).clone()}

    def get_observations(self):
        """캐시된 현재 관측과 이력을 반환하며 통계나 이력을 갱신하지 않는다."""
        return self._current

    def reset(self):
        """새 episode의 최초 관측으로 이력 전체를 채운다."""
        obs_dict, _ = self._env.reset()
        self._history.copy_(obs_dict["policy_current"].unsqueeze(1))
        self._critic_history.copy_(obs_dict["critic"].unsqueeze(1))
        self._current = self._pack(obs_dict)
        return self._current

    def step(self, actions):
        """현재 전이의 다음 관측·종료 상태·terminal critic을 반환한다."""
        obs_dict, reward, terminated, truncated, extras = self._env.step(actions)
        done = terminated | truncated
        # 종료 직전 단일 관측을 기존 이력 뒤에 붙여 timeout bootstrap을 보존한다.
        terminal_critic = torch.cat((self._critic_history[:, 1:],
                                     extras["terminal_critic"].unsqueeze(1)), 1).flatten(1)
        # critic 특권 관측 이력과 actor 부분 관측 이력을 각각 갱신하고 종료 환경은 새 관측으로 채운다.
        self._critic_history = torch.cat((self._critic_history[:, 1:],
                                          obs_dict["critic"].unsqueeze(1)), 1)
        self._critic_history[done] = obs_dict["critic"][done].unsqueeze(1)
        self._history = torch.cat((self._history[:, 1:], obs_dict["policy_current"].unsqueeze(1)), 1)
        self._history[done] = obs_dict["policy_current"][done].unsqueeze(1)
        self._current = self._pack(obs_dict)
        # 복원 대상인 다음 관측과, 종료되지 않은 전이만 참인 복원 손실 유효 mask를 함께 반환한다.
        result = {
            "rewards": reward, "terminated": terminated, "truncated": truncated,
            "prediction_valid": ~done, "next_obs": obs_dict["policy_current"],
            "terminal_critic": terminal_critic,
            "terminal_valid": extras["terminal_valid"],
        }

        # 환경이 남긴 manager 로그와 진단값을 학습 기록용으로 함께 전달한다.
        result["log"] = extras.get("log", {})
        result["diagnostics"] = extras.get("diagnostics", {})

        return self._current, result
