# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin


"""DreamWaQ rollout 수집·학습·체크포인트를 관리한다."""
import copy
import logging
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from ..modules.actor_critic_DWAQ import ActorCritic_DWAQ
from ..algorithms.ppo import PPO
from ..algorithms.adaboot import AdaBoot
from ..storage.rollout_storage import RolloutStorage

log = logging.getLogger(__name__)

# 이전 버전(rsl_rl 스타일) 배너 로그의 폭·라벨 정렬 칸수를 그대로 재사용한다.
_LOG_WIDTH = 80
_LOG_PAD = 35


def _format_iteration_log(iteration, total_iterations, metrics, eta_seconds):
    """구분선·중앙정렬 제목·우측정렬 라벨의 이전 배너 양식으로 iteration 로그를 만든다."""
    title = f" Learning iteration {iteration}/{total_iterations} "
    lines = ["#" * _LOG_WIDTH, title.center(_LOG_WIDTH, " "), ""]
    lines += [f"{key + ':':>{_LOG_PAD}} {value:.6g}" for key, value in metrics.items()]
    lines += ["-" * _LOG_WIDTH, f"{'ETA:':>{_LOG_PAD}} {eta_seconds:.1f}s"]
    return "\n".join(lines)


class OnPolicyRunner:
    """시점·정규화·AdaBoot 조건을 보존하는 학습 루프."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        """환경 계약으로 모델을 구성하고 설정을 보존한다."""
        self._env = env
        self._cfg = copy.deepcopy(train_cfg)
        self._device = torch.device(device)
        if torch.device(env.device) != self._device:
            raise ValueError("환경과 학습 장치는 동일해야 합니다.")
        runner_cfg = self._cfg["runner"]
        if runner_cfg["policy_class_name"] != "ActorCritic_DWAQ" or runner_cfg["algorithm_class_name"] != "PPO":
            raise ValueError("이 runner는 DreamWaQ/PPO 전용입니다.")
        self._model = ActorCritic_DWAQ(
            env.num_obs, env.num_privileged_obs, env.num_actions,
            env.actor_history_length, env.estimator_history_length,
            **self._cfg["policy"]).to(self._device)
        self._algorithm = PPO(self._model, **self._cfg["algorithm"])
        self._adaboot = AdaBoot(env.num_envs, self._device, **self._cfg["adaboot"])
        self._steps = runner_cfg["num_steps_per_env"]
        self._save_interval = runner_cfg["save_interval"]
        self._storage = RolloutStorage(env.num_envs, self._steps, env.num_obs,
                                       env.num_privileged_obs,
                                       env.num_obs * env.actor_history_length,
                                       env.num_actions, self._device)
        self._log_dir = Path(log_dir) if log_dir is not None else None
        self._phase = int(env.training_phase)
        self._iteration = 0
        self._normalization_initialized = False

    @property
    def phase(self):
        """현재 학습 단계 번호를 반환한다."""
        return self._phase

    @property
    def model(self):
        """평가 지표 계산에 사용할 모델을 반환한다."""
        return self._model

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        """rollout와 PPO update 동안 정규화 통계·AdaBoot 확률을 고정한다."""
        observation = self._env.get_observations()
        if init_at_random_ep_len:
            self._env.episode_length_buf.copy_(torch.randint_like(
                self._env.episode_length_buf, high=self._env.max_episode_length))
        if not self._normalization_initialized:
            self._model.update_normalizers(observation["obs"], observation["critic"], observation["velocity"])
            self._normalization_initialized = True
        episode_return = torch.zeros(self._env.num_envs, device=self._device)
        writer = SummaryWriter(str(self._log_dir)) if self._log_dir is not None else None
        self._model.train()
        total_iterations = self._iteration + num_learning_iterations
        learn_start_time = time.time()
        try:
            for local_iteration in range(num_learning_iterations):
                probability = self._adaboot.probability()
                completed_returns = []
                velocity_error = torch.zeros((), device=self._device)
                with torch.no_grad():
                    for _ in range(self._steps):
                        use_estimate = torch.rand(self._env.num_envs, device=self._device) < probability
                        policy = self._model.distribution(observation["history"],
                                                          observation["velocity"], use_estimate)
                        actions = policy.sample()
                        transition = {
                            "obs": observation["obs"], "critic": observation["critic"],
                            "history": observation["history"], "velocity": observation["velocity"],
                            "actions": actions, "use_estimate": use_estimate,
                            "mean": policy.mean, "std": policy.scale,
                            "log_prob": policy.log_prob(actions).sum(-1),
                            "values": self._model.evaluate(observation["critic"]),
                        }
                        velocity_error += (self._model.estimate_velocity(observation["history"])
                                           - observation["velocity"]).square().mean()
                        observation, result = self._env.step(actions)
                        done = result["terminated"] | result["truncated"]
                        timeout = result["truncated"] & ~result["terminated"]
                        if (timeout & ~result["terminal_valid"]).any():
                            raise RuntimeError("시간 제한 종료의 terminal critic이 없습니다.")
                        next_critic = torch.where(timeout.unsqueeze(-1), result["terminal_critic"],
                                                  observation["critic"])
                        transition.update({key: result[key] for key in (
                            "rewards", "terminated", "truncated", "prediction_valid", "next_obs")})
                        transition["next_values"] = self._model.evaluate(next_critic)
                        self._storage.add(transition)
                        episode_return += result["rewards"]
                        self._adaboot.update(episode_return, done)
                        completed_returns.append(episode_return[done].clone())
                        episode_return[done] = 0
                self._storage.compute_returns(self._algorithm.gamma, self._algorithm.lam)
                metrics = self._algorithm.update(self._storage)
                metrics["adaboot_probability"] = probability.item()
                metrics["velocity_rmse_mps"] = (velocity_error / self._steps).sqrt().item()
                completed = torch.cat(completed_returns)
                if completed.numel():
                    metrics["episode_return"] = completed.mean().item()
                self._model.update_normalizers(*self._storage.normalization_data())
                self._storage.clear()
                self._iteration += 1
                for key, value in metrics.items():
                    if writer is not None:
                        writer.add_scalar(key, value, self._iteration)
                elapsed = time.time() - learn_start_time
                remaining = num_learning_iterations - (local_iteration + 1)
                eta_seconds = elapsed / (local_iteration + 1) * remaining
                log.info(_format_iteration_log(self._iteration, total_iterations, metrics, eta_seconds))
                if self._log_dir is not None and self._iteration % self._save_interval == 0:
                    self.save(self._log_dir / f"model_p{self._phase}_{self._iteration}.pt")
            if self._log_dir is not None:
                self.save(self._log_dir / f"model_p{self._phase}_{self._iteration}.pt")
        finally:
            if writer is not None:
                writer.close()

    def save(self, path):
        """모델 통계와 입력 규격·실제 반복 횟수를 함께 저장한다."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "specification": self._env.specification,
            "train_cfg": self._cfg,
            "environment_configuration": self._env.training_configuration,
            "curriculum": self._env.curriculum_state(),
            "phase": self._phase,
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._algorithm.state_dict(),
            "adaboot": self._adaboot.state_dict(),
            "iteration": self._iteration,
            "normalization_initialized": self._normalization_initialized,
        }, path)

    def load(self, path, mode="resume"):
        """재개·단계 전환·평가를 구분해 체크포인트를 복원한다.

        resume은 같은 단계의 학습을 그대로 이어가므로 학습 설정과 보상·명령 설정까지
        완전히 같아야 한다. transfer는 1단계 정책을 2단계로 넘기는 경로이며,
        관측·제어 규격은 동일해야 하지만 보상과 지형 설정의 차이는 허용한다.
        """
        if mode not in ("resume", "transfer", "evaluate"):
            raise ValueError(f"지원하지 않는 checkpoint 사용 방식: {mode}")
        checkpoint = torch.load(path, map_location=self._device, weights_only=True)
        if checkpoint["specification"] != self._env.specification:
            raise ValueError("체크포인트의 관절·관측·action 규격이 현재 환경과 다릅니다.")
        if checkpoint["train_cfg"]["policy"] != self._cfg["policy"]:
            raise ValueError("체크포인트와 정책/정규화 설정이 다릅니다.")
        if mode == "resume":
            if checkpoint["phase"] != self._phase:
                raise ValueError("다른 단계의 체크포인트입니다. 단계 전환에는 transfer를 사용하세요.")
            if checkpoint["train_cfg"] != self._cfg:
                raise ValueError("학습 재개에는 저장된 학습 설정을 사용해야 합니다.")
            if checkpoint["environment_configuration"] != self._env.training_configuration:
                raise ValueError("학습 재개 시 보상·관측·랜덤화·명령 설정이 다릅니다.")
        if mode == "transfer" and checkpoint["phase"] + 1 != self._phase:
            raise ValueError("transfer는 직전 단계의 체크포인트에서만 이어받을 수 있습니다.")

        # 정규화 통계는 모델 버퍼에 있으므로 세 방식 모두 그대로 이어받는다.
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._normalization_initialized = checkpoint["normalization_initialized"]
        if mode == "resume":
            self._algorithm.load_state_dict(checkpoint["optimizer_state_dict"])
            self._adaboot.load_state_dict(checkpoint["adaboot"])
            self._env.load_curriculum_state(checkpoint["curriculum"])
            self._iteration = checkpoint["iteration"]
        elif mode == "transfer":
            # 보상 구성이 바뀌어 gradient·return 스케일이 달라지므로
            # optimizer 모멘텀과 AdaBoot return 통계는 이어받지 않는다.
            self._env.load_curriculum_state(checkpoint["curriculum"])
            self._iteration = 0
        else:
            self._iteration = checkpoint["iteration"]
        log.info("checkpoint=%s mode=%s phase=%d iteration=%d loaded",
                 path, mode, self._phase, self._iteration)

    def get_inference_policy(self, device=None):
        """정규화 갱신이나 실제 속도 입력 없이 결정적 정책을 반환한다."""
        if device is not None and torch.device(device) != self._device:
            raise ValueError("평가 장치는 runner 생성 시 지정해야 합니다.")
        self._model.eval()
        return self._model.act_inference
