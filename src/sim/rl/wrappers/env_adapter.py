"""Isaac Lab 환경 관측을 DreamWaQ 학습 입력(현재 관측·관측 이력·특권 관측·속도 정답)으로 묶는다."""
import torch

from src.controller.legged import HISTORY_LENGTH


class EnvironmentAdapter:
    """CENet 입력용 관측 이력을 관리하고 전이 결과를 학습기에 전달한다."""

    def __init__(self, env):
        """환경을 reset하고 관측 차원과 관측 이력 버퍼를 준비한다."""
        self._env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.step_dt = env.step_dt
        self.max_episode_length = env.max_episode_length
        self.history_length = HISTORY_LENGTH
        self.num_actions = env.action_manager.total_action_dim
        obs_dict, _ = env.reset()
        self.num_obs = obs_dict["policy"].shape[-1]
        self.num_critic_obs = obs_dict["critic"].shape[-1]
        self._history = obs_dict["policy"].unsqueeze(1).repeat(1, HISTORY_LENGTH, 1)
        self._current = self._pack(obs_dict)

    @property
    def specification(self):
        """실행 기록용 관측·행동 규격을 반환한다."""
        return {"obs_dim": self.num_obs, "critic_dim": self.num_critic_obs, "actions": self.num_actions,
                "history_length": self.history_length, "step_dt": self.step_dt,
                "action_joint_names": list(self._env.action_manager.get_term("joint_pos").joint_names)}

    @property
    def terrain_level(self):
        """전체 환경의 평균 지형 curriculum 레벨을 반환한다. 지형 curriculum이 없으면 None이다."""
        terrain_levels = getattr(self._env.scene.terrain, "terrain_levels", None)
        if terrain_levels is None:
            return None
        return terrain_levels.float().mean()

    def _pack(self, obs_dict):
        """현재 관측·관측 이력(오래된 프레임부터)·특권 관측·속도 정답을 묶는다."""
        return {"obs": obs_dict["policy"], "history": self._history.flatten(1).clone(),
                "critic": obs_dict["critic"], "velocity": obs_dict["velocity_target"]}

    def get_observations(self):
        """현재 관측 묶음을 반환한다."""
        return self._current

    def reset(self):
        """환경을 reset하고 관측 이력을 최초 관측으로 채운다."""
        obs_dict, _ = self._env.reset()
        self._history.copy_(obs_dict["policy"].unsqueeze(1))
        self._current = self._pack(obs_dict)
        return self._current

    def step(self, actions):
        """환경을 한 스텝 진행하고 다음 관측 묶음과 보상·종료·다음 관측·로그를 반환한다."""
        obs_dict, rewards, terminated, truncated, extras = self._env.step(actions)
        dones = terminated | truncated

        # 관측 이력에 새 관측을 붙이고, 종료된 환경은 새 episode 첫 관측으로 채운다.
        self._history = torch.cat((self._history[:, 1:], obs_dict["policy"].unsqueeze(1)), 1)
        self._history[dones] = obs_dict["policy"][dones].unsqueeze(1)
        self._current = self._pack(obs_dict)
        # 전이 결과와 평가용 명령·몸통 yaw rate를 묶는다.
        result = {"rewards": rewards, "dones": dones, "time_outs": truncated & ~terminated,
                  "terminated": terminated, "next_obs": obs_dict["policy"], "log": extras.get("log", {}),
                  "command": self._env.command_manager.get_command("base_velocity"),
                  "yaw_rate": self._env.scene["robot"].data.root_ang_vel_b[:, 2]}
        return self._current, result
