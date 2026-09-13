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

from ..env import VecEnv
from ..modules.actor_critic_DWAQ import ActorCritic_DWAQ
from ..algorithms.ppo import PPO
from ..algorithms.adaboot import AdaBoot
from ..storage.rollout_storage import RolloutStorage

log = logging.getLogger(__name__)



def _format_iteration_log(iteration, total_iterations, metrics, eta_seconds):
    """콘솔에는 보행 성능과 최적화 상태를 요약하고 상세 항목은 TensorBoard에 남긴다."""
    fields = (
        ("episode_return", "Return"),
        ("Episodes/mean_length_s", "Episode s"),
        ("Episodes/timeout_fraction", "Timeout fraction"),
        ("Tracking/linear_error_mps", "Velocity error m/s"),
        ("Tracking/yaw_error_radps", "Yaw error rad/s"),
        ("Control/requested_target_clipped_fraction", "Target clip fraction"),
        ("value", "Value loss"),
        ("policy_kl", "KL"),
        ("learning_rate", "LR"),
    )
    title = f" Learning iteration {iteration}/{total_iterations} "
    lines = ["#" * 80, title.center(80), ""]
    lines.extend(f"{label + ':':>35} {metrics[key]:.6g}" for key, label in fields if key in metrics)
    lines.extend(["-" * 80, f"{'ETA:':>35} {eta_seconds:.1f}s"])
    return "\n".join(lines)



class OnPolicyRunner:
    """시점·정규화·AdaBoot 조건을 보존하는 학습 루프.

    teacher-student 없이 actor·critic·추정기를 하나의 학습 루프에서 함께 학습한다.
    """

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        """환경 계약으로 모델을 구성하고 설정을 보존한다."""
        self._env = env
        self._cfg = copy.deepcopy(train_cfg)
        self._device = torch.device(device)
        if torch.device(env.device) != self._device:
            raise ValueError("환경과 학습 장치는 동일해야 합니다.")
        runner_cfg = self._cfg["runner"]
        if runner_cfg["policy_class_name"] != "ActorCritic_DWAQ" or runner_cfg["algorithm_class_name"] != "PPO":
            raise ValueError("이 runner는 DreamWaQ/PPO 전용입니다.")
        # 정책·가치·추정기를 하나의 모델로 구성하고 PPO 하나로 동시에 최적화한다.
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

        # 학습 기록용 진행 중 episode의 return과 길이를 환경별로 누적한다.
        self._episode_return = torch.zeros(env.num_envs, device=self._device)
        self._episode_steps = torch.zeros(env.num_envs, device=self._device)

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
        # 현재 관측을 받고 필요하면 episode 길이를 무작위로 초기화한다.
        observation = self._env.get_observations()
        if init_at_random_ep_len:
            self._env.episode_length_buf.copy_(torch.randint_like(
                self._env.episode_length_buf, high=self._env.max_episode_length))

        # 첫 학습이면 초기 관측으로 정규화 통계를 만들고 모델을 학습 모드로 둔다.
        if not self._normalization_initialized:
            self._model.update_normalizers(observation["obs"], observation["critic"])
            self._normalization_initialized = True
        self._model.train()

        # 학습 루프 전에 TensorBoard writer, 남은 시간 계산 기준, rollout 통계 누적기를 준비한다.
        writer = SummaryWriter(str(self._log_dir)) if self._log_dir is not None else None
        total_iterations = self._iteration + num_learning_iterations
        learn_start_time = time.time()
        completed_returns = []
        completed_lengths = []
        diagnostics = {}
        manager_logs = {}
        manager_samples = torch.zeros((), device=self._device)
        deaths = torch.zeros((), device=self._device)
        timeouts = torch.zeros((), device=self._device)
        velocity_error = torch.zeros((), device=self._device)

        try:
            for local_iteration in range(num_learning_iterations):
                # iteration마다 AdaBoot 확률을 한 번 계산해 rollout 동안 고정한다.
                probability = self._adaboot.probability()

                with torch.no_grad():
                    for _ in range(self._steps):
                        # 환경별로 AdaBoot 확률에 따라 추정 속도와 실제 속도 중 정책 입력을 고른다.
                        use_estimate = torch.rand(self._env.num_envs, device=self._device) < probability
                        # actor는 부분 관측 이력으로 행동을 샘플링한다.
                        policy = self._model.distribution(observation["history"],
                                                          observation["velocity"], use_estimate)
                        actions = policy.sample()
                        # critic 가치는 특권 관측으로만 계산해 전이에 함께 저장한다.
                        transition = {
                            "obs": observation["obs"], "critic": observation["critic"],
                            "history": observation["history"], "velocity": observation["velocity"],
                            "actions": actions, "use_estimate": use_estimate,
                            "mean": policy.mean, "std": policy.scale,
                            "log_prob": policy.log_prob(actions).sum(-1),
                            "values": self._model.evaluate(observation["critic"]),
                        }

                        # 환경을 한 스텝 진행하고 timeout은 terminal critic으로 다음 가치를 계산해 전이를 저장한다.
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
                        # AdaBoot에 스텝 보상과 종료 여부를 전달해 환경별 episode return을 갱신한다.
                        self._adaboot.update(result["rewards"], done)

                        # 행동 전 관측의 추정 속도 오차, 완료 episode 길이·return, 종료 사유, 진단값, manager 로그를 누적한다.
                        velocity_error += (self._model.estimate_velocity(transition["history"])
                                           - transition["velocity"]).square().mean()
                        self._episode_steps += 1
                        completed_lengths.append(self._episode_steps[done].clone())
                        self._episode_steps[done] = 0
                        self._episode_return += result["rewards"]
                        completed_returns.append(self._episode_return[done].clone())
                        self._episode_return[done] = 0
                        deaths += result["terminated"].sum()
                        timeouts += timeout.sum()
                        for key, value in result.get("diagnostics", {}).items():
                            diagnostics[key] = diagnostics.get(key, 0.0) + value.detach()
                        count = done.sum()
                        if result.get("log"):
                            manager_samples += count
                            for key, value in result["log"].items():
                                value = torch.as_tensor(value, device=self._device).float().mean()
                                manager_logs[key] = manager_logs.get(key, 0.0) + value * count

                # 수집한 rollout 하나로 actor·critic·추정기를 teacher 없이 한 번에 갱신한다.
                self._storage.compute_returns(self._algorithm.gamma, self._algorithm.lam)
                metrics = self._algorithm.update(self._storage)

                # 사용한 rollout으로 정규화 통계를 갱신하고 다음 rollout을 준비한다.
                self._model.update_normalizers(*self._storage.normalization_data())
                self._storage.clear()
                self._iteration += 1

                # 저장 주기마다 체크포인트를 저장한다.
                if self._log_dir is not None and self._iteration % self._save_interval == 0:
                    self.save(self._log_dir / f"model_p{self._phase}_{self._iteration}.pt")

                # rollout 통계·curriculum 지표를 학습 지표에 합쳐 TensorBoard와 터미널에 기록하고 누적기를 비운다.
                for key, value in diagnostics.items():
                    divisor = 1 if key.startswith("TerminationCount/") else self._steps
                    metrics[key] = (value / divisor).item()
                for key, value in manager_logs.items():
                    metrics["Manager/" + key] = (value / manager_samples.clamp_min(1)).item()
                lengths = torch.cat(completed_lengths)
                metrics["Episodes/completed"] = lengths.numel()
                metrics["Episodes/terminated"] = deaths.item()
                metrics["Episodes/timeouts"] = timeouts.item()
                if lengths.numel():
                    metrics["Episodes/mean_length_s"] = (lengths.mean() * self._env.step_dt).item()
                    metrics["Episodes/timeout_fraction"] = (timeouts / lengths.numel()).item()
                completed = torch.cat(completed_returns)
                if completed.numel():
                    metrics["episode_return"] = completed.mean().item()
                metrics["adaboot_probability"] = probability.item()
                metrics["velocity_rmse_mps"] = (velocity_error / self._steps).sqrt().item()
                curriculum = self._env.curriculum_metrics()
                if curriculum:
                    values = torch.stack(list(curriculum.values())).tolist()
                    metrics.update(zip(curriculum.keys(), values))
                for key, value in metrics.items():
                    if writer is not None:
                        writer.add_scalar(key, value, self._iteration)
                elapsed = time.time() - learn_start_time
                remaining = num_learning_iterations - (local_iteration + 1)
                eta_seconds = elapsed / (local_iteration + 1) * remaining
                log.info(_format_iteration_log(self._iteration, total_iterations, metrics, eta_seconds))
                completed_returns.clear()
                completed_lengths.clear()
                diagnostics.clear()
                manager_logs.clear()
                manager_samples.zero_()
                deaths.zero_()
                timeouts.zero_()
                velocity_error.zero_()

            # 학습이 끝나면 마지막 체크포인트를 저장한다.
            if self._log_dir is not None:
                self.save(self._log_dir / f"model_p{self._phase}_{self._iteration}.pt")
        finally:
            # TensorBoard 기록 파일을 닫는다.
            if writer is not None:
                writer.close()

    def save(self, path):
        """모델 통계와 입력 규격·실제 반복 횟수를 함께 저장한다."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "train_cfg": self._cfg,
            "curriculum": self._env.curriculum_state(),
            "phase": self._phase,
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._algorithm.state_dict(),
            "adaboot": self._adaboot.state_dict(),
            "iteration": self._iteration,
            "normalization_initialized": self._normalization_initialized,
        }, path)

    def load(self, path, mode="resume"):
        """학습 재개 또는 평가 방식에 따라 모델과 학습 상태를 복원한다.

        resume은 checkpoint 단계가 현재 단계와 같으면 학습 상태 전체를 복원하고,
        다르면 모델과 명령 curriculum만 이어받아 새 단계 학습을 시작한다.
        """
        if mode not in ("resume", "evaluate"):
            raise ValueError(f"지원하지 않는 checkpoint 사용 방식: {mode}")
        checkpoint = torch.load(path, map_location=self._device, weights_only=True)

        # 모델 가중치와 모델 버퍼에 있는 관측 정규화 통계를 복원한다.
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._normalization_initialized = checkpoint["normalization_initialized"]

        same_phase = checkpoint["phase"] == self._phase
        if mode == "resume" and same_phase:
            # 같은 단계는 optimizer·AdaBoot 통계·명령 curriculum·반복 횟수를 모두 복원한다.
            self._algorithm.load_state_dict(checkpoint["optimizer_state_dict"])
            self._adaboot.load_state_dict(checkpoint["adaboot"])
            self._env.load_curriculum_state(checkpoint["curriculum"])
            self._iteration = checkpoint["iteration"]
        elif mode == "resume":
            # 다른 단계는 명령 curriculum만 복원하고 optimizer·AdaBoot·반복 횟수는 새로 시작한다.
            self._env.load_curriculum_state(checkpoint["curriculum"])
            self._iteration = 0
        else:
            # 평가는 반복 횟수만 기록용으로 복원한다.
            self._iteration = checkpoint["iteration"]

        # 진행 중 episode의 return·길이 기록을 초기화하고 적재 결과를 터미널에 기록한다.
        self._episode_return.zero_()
        self._episode_steps.zero_()
        log.info("checkpoint=%s mode=%s checkpoint_phase=%d phase=%d iteration=%d loaded",
                 path, mode, checkpoint["phase"], self._phase, self._iteration)

    def get_inference_policy(self, device=None):
        """정규화 갱신이나 실제 속도 입력 없이 결정적 정책을 반환한다."""
        if device is not None and torch.device(device) != self._device:
            raise ValueError("평가 장치는 runner 생성 시 지정해야 합니다.")
        self._model.eval()
        return self._model.act_inference
