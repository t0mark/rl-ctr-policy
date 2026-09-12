"""Isaac Lab 관측을 시점이 명시된 DreamWaQ 전이로 변환한다."""
import copy
import hashlib
import inspect
import torch


def configuration_value(value):
    """설정 객체를 재현 가능한 기본 자료형으로 변환한다."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, slice):
        return {"slice": [value.start, value.stop, value.step]}
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return [configuration_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): configuration_value(item) for key, item in value.items()}
    if callable(value):
        target = value if hasattr(value, "__qualname__") else type(value)
        return target.__module__ + "." + target.__qualname__
    if hasattr(value, "to_dict"):
        return configuration_value(value.to_dict())
    if hasattr(value, "__dict__"):
        return {key: configuration_value(item) for key, item in vars(value).items() if not key.startswith("_")}
    raise TypeError(f"저장할 수 없는 환경 설정 자료형: {type(value)}")


def observation_contract(env):
    """평가 노이즈 스위치를 제외한 관측 함수·전처리 계약을 기록한다."""
    contract = {}
    for group_name, names in env.observation_manager.active_terms.items():
        group = getattr(env.cfg.observations, group_name)
        serialized = configuration_value(group)
        serialized.pop("enable_corruption", None)
        for name in names:
            term = getattr(group, name)
            serialized[name].pop("noise", None)
            target = term.func if inspect.isfunction(term.func) else type(term.func)
            source = inspect.getsource(target)
            serialized[name]["source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
        contract[group_name] = serialized
    return contract


class DwaqVecEnvWrapper:
    """현재 관측을 한 번만 읽어 같은 표본으로 이력을 구성한다."""

    def __init__(self, env):
        """환경을 초기화하고 실제 관측 차원과 배열 순서를 기록한다."""
        self._env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.num_actions = env.action_manager.total_action_dim
        self.max_episode_length = env.max_episode_length
        self.num_obs_hist = env.cfg.dwaq_history_length
        obs_dict, _ = env.reset()
        self.num_obs = obs_dict["policy_current"].shape[-1]
        self.num_privileged_obs = obs_dict["critic"].shape[-1]
        self._history = obs_dict["policy_current"].unsqueeze(1).repeat(1, self.num_obs_hist, 1)
        self._current = self._pack(obs_dict)
        robot = env.scene["robot"]
        action_term = env.action_manager.get_term("joint_pos")
        self._training_configuration = {
            key: configuration_value(getattr(env.cfg, key, None))
            for key in ("events", "rewards", "commands", "curriculum", "observations", "actions")
        }
        defaults = robot.data.default_joint_pos
        if not torch.equal(defaults, defaults[:1].expand_as(defaults)):
            raise ValueError("checkpoint 계약에는 모든 환경에서 동일한 기본 관절 자세가 필요합니다.")
        self._spec = {
            "default_joint_pos": configuration_value(defaults[0]),
            "observation_contract": observation_contract(env),
            "obs_dim": self.num_obs, "critic_dim": self.num_privileged_obs,
            "actions": self.num_actions, "history_length": self.num_obs_hist,
            "history_order": "oldest_to_current_frames",
            "joint_names": list(robot.joint_names),
            "action_joint_names": list(action_term.joint_names),
            "policy_terms": list(env.observation_manager.active_terms["policy_current"]),
            "critic_terms": list(env.observation_manager.active_terms["critic"]),
            "step_dt": env.step_dt,
            "action_scale": env.cfg.actions.joint_pos.scale,
            "actuators": configuration_value(getattr(getattr(env.cfg, "scene", None), "robot", None).actuators)
                if hasattr(getattr(getattr(env.cfg, "scene", None), "robot", None), "actuators") else None,
        }

    @property
    def specification(self):
        """체크포인트와 관측·제어 계약을 비교할 메타데이터를 제공한다."""
        return copy.deepcopy(self._spec)

    @property
    def training_configuration(self):
        """학습 재개 시 보상·랜덤화·명령 설정의 동등성을 확인한다."""
        return self._training_configuration

    def curriculum_state(self):
        """명령 curriculum에서 이미 열린 셀을 저장한다."""
        manager = getattr(self._env, "command_manager", None)
        if manager is None:
            return None
        return manager.get_term("base_velocity").curriculum_state()

    def load_curriculum_state(self, state):
        """새 episode를 시작하기 전에 명령 curriculum을 복원한다."""
        if state is not None:
            self._env.command_manager.get_term("base_velocity").load_curriculum_state(state)
        self.reset()

    @property
    def episode_length_buf(self):
        """환경의 episode 길이를 노출한다."""
        return self._env.episode_length_buf

    def _pack(self, obs_dict):
        """속도 타깃을 critic 슬라이싱 없이 별도 관측 그룹에서 얻는다."""
        return {"obs": obs_dict["policy_current"], "critic": obs_dict["critic"],
                "velocity": obs_dict["velocity_target"],
                "history": self._history.flatten(1).clone()}

    def get_observations(self):
        """캐시된 현재 관측과 이력을 반환하며 통계나 이력을 갱신하지 않는다."""
        return self._current

    def reset(self):
        """새 episode의 최초 관측으로 이력 전체를 채운다."""
        obs_dict, _ = self._env.reset()
        self._history.copy_(obs_dict["policy_current"].unsqueeze(1))
        self._current = self._pack(obs_dict)
        return self._current

    def step(self, actions):
        """현재 전이의 다음 관측·종료 상태·terminal critic을 반환한다."""
        obs_dict, reward, terminated, truncated, extras = self._env.step(actions)
        done = terminated | truncated
        self._history = torch.cat((self._history[:, 1:], obs_dict["policy_current"].unsqueeze(1)), 1)
        self._history[done] = obs_dict["policy_current"][done].unsqueeze(1)
        self._current = self._pack(obs_dict)
        return self._current, {
            "rewards": reward, "terminated": terminated, "truncated": truncated,
            "prediction_valid": ~done, "next_obs": obs_dict["policy_current"],
            "terminal_critic": extras["terminal_critic"],
            "terminal_valid": extras["terminal_valid"],
            "log": extras.get("log", {}),
        }
