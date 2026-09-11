# Copyright (c) 2026 DreamWaQ Project
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab ManagerBasedRLEnv <-> dwaq(PPO/OnPolicyRunner) 인터페이스 어댑터.

dwaq(src/dwaq)는 시뮬레이터를 모르는 학습 라이브러리로, env가 특정 계약(속성/메서드)을
만족한다고 가정하고 동작한다(src/dwaq/env/vec_env.py의 VecEnv, 그리고 실제로는 이를 더
구체화한 레거시 BaseTask/LeggedRobot의 계약 — reset()/step()이 4-tuple/7-tuple을 반환하고,
관측이 "현재 스텝(obs)"과 "flatten된 이력(obs_hist)"으로 분리돼 있는 것 등).

Isaac Lab의 ManagerBasedRLEnv는 이 계약과 무관한 자기 자신의 인터페이스
(step()이 (obs_dict, reward, terminated, truncated, extras) 5-tuple을 반환)를 갖고 있어서,
이 클래스가 둘 사이를 그대로 이어주는 얇은 런타임 어댑터 역할을 한다. Isaac Lab이 자기
진영(vanilla rsl_rl)을 위해 제공하는 공식 RslRlVecEnvWrapper와 동일한 역할이며, 변환
방식(dones 계산, time_outs/log 처리)도 그 공식 구현을 그대로 따른다.
"""

import torch
from isaaclab.envs import ManagerBasedRLEnv


class DwaqVecEnvWrapper:
    """ManagerBasedRLEnv 인스턴스 하나를 감싸서 dwaq가 기대하는 env 계약으로 변환한다.

    관측은 env_cfg.py의 ObservationsCfg가 정의한 세 그룹으로 분리해서 가져온다.
        policy_current: 현재 스텝 고유수용감각 -> dwaq의 "obs" (actor 입력)
        policy        : 위 관측의 이력(history) -> dwaq의 "obs_hist" (컨텍스트 인코더 입력)
        critic        : 특권 정보 포함 관측 -> dwaq의 "privileged_obs" (critic 입력)
    """

    def __init__(self, env: ManagerBasedRLEnv):
        """env를 감싸고, reset을 한 번 실행해 관측 차원(num_obs 등)을 실측으로 확정한다."""
        self._env = env

        self.num_envs = env.num_envs
        self.device = env.device
        self.max_episode_length = env.max_episode_length
        self.num_actions = env.action_manager.total_action_dim
        # 컨텍스트 인코더(CENet) 입력 차원 계산에 쓰이는 이력 길이(dwaq의 num_obs_hist).
        self.num_obs_hist = env.cfg.observations.policy.history_length

        # 관측 차원은 Isaac Lab 내부 API 이름을 추측하지 않고, 실제 reset() 결과의
        # 텐서 shape으로 직접 확정한다.
        obs_dict, _ = env.reset()
        self.num_obs = obs_dict["policy_current"].shape[-1]
        self.num_privileged_obs = obs_dict["critic"].shape[-1]

        # PolicyCfg(이력)와 PolicyCurrentCfg(현재 스텝)의 관측 항목 구성이 어긋나면
        # 이력 차원이 (이력 길이 * 현재 스텝 차원)과 달라진다 — 즉시 감지되도록 검증한다.
        obs_hist_dim = obs_dict["policy"].shape[-1]
        assert obs_hist_dim == self.num_obs_hist * self.num_obs, (
            f"policy(이력) 관측 차원({obs_hist_dim})이 history_length({self.num_obs_hist}) * "
            f"policy_current 차원({self.num_obs})과 다릅니다. "
            "PolicyCfg와 PolicyCurrentCfg의 관측 항목 구성이 일치하는지 확인하세요."
        )

        # dwaq PPO는 이전 스텝의 privileged_obs를 CENet 속도 추정 학습의 정답으로 쓴다.
        self._prev_privileged_obs = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device)

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """현재 에피소드 진행 스텝 수. dwaq 러너가 랜덤 초기화를 위해 직접 덮어쓰기도 한다."""
        return self._env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        self._env.episode_length_buf = value

    def get_observations(self):
        """(현재 스텝 관측, 관측 이력) 튜플을 새로 계산해서 반환한다."""
        obs_dict = self._env.observation_manager.compute()
        return obs_dict["policy_current"], obs_dict["policy"]

    def get_privileged_observations(self):
        """(특권 관측, 직전 스텝 특권 관측) 튜플을 반환한다."""
        obs_dict = self._env.observation_manager.compute()
        return obs_dict["critic"], self._prev_privileged_obs

    def reset(self):
        """환경을 리셋하고 (obs, privileged_obs, prev_privileged_obs, obs_hist)를 반환한다."""
        obs_dict, _ = self._env.reset()
        privileged_obs = obs_dict["critic"]
        prev_privileged_obs = torch.zeros_like(privileged_obs)
        self._prev_privileged_obs = privileged_obs.clone()
        return obs_dict["policy_current"], privileged_obs, prev_privileged_obs, obs_dict["policy"]

    def step(self, actions: torch.Tensor):
        """한 스텝 진행하고 (obs, privileged_obs, prev_privileged_obs, obs_hist, rewards, dones, infos)를 반환한다."""
        obs_dict, rewards, terminated, truncated, extras = self._env.step(actions)

        privileged_obs = obs_dict["critic"]
        prev_privileged_obs = self._prev_privileged_obs
        self._prev_privileged_obs = privileged_obs.clone()

        # dones: Isaac Lab 공식 RslRlVecEnvWrapper와 동일하게 종료+시간초과를 합친다.
        dones = (terminated | truncated).to(dtype=torch.long)

        # 시간초과(time out) 정보: dwaq PPO가 부트스트래핑에 쓰는 키로 옮겨 담는다.
        extras["time_outs"] = truncated
        # 에피소드 로그: Isaac Lab은 "log" 키를 쓰지만, dwaq 러너는 "episode" 키를 읽는다.
        if "log" in extras:
            extras["episode"] = extras.pop("log")

        return (
            obs_dict["policy_current"],
            privileged_obs,
            prev_privileged_obs,
            obs_dict["policy"],
            rewards,
            dones,
            extras,
        )
