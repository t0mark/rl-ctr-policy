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


"""DreamWaQ 정책의 rollout 수집·학습·평가를 관리하는 on-policy runner."""
import copy
import logging
import time

import torch

from src.controller.legged import ActorCritic
from src.sim.rl.algorithms.adaboot import AdaBoot
from src.sim.rl.algorithms.ppo import GAE_LAMBDA, GAMMA, PPO
from src.sim.rl.storage.rollout_storage import RolloutStorage

log = logging.getLogger(__name__)

# iteration마다 환경별로 수집하는 rollout 스텝 수.
NUM_STEPS_PER_ENV = 24

# 평균 지형 레벨이 가장 높았던 정책을 저장하는 checkpoint 파일명.
BEST_CHECKPOINT_NAME = "model_best.pt"

# 터미널에 요약하는 학습 지표와 표시 이름. 전체 지표는 TensorBoard에 기록한다.
_CONSOLE_FIELDS = (
    ("episode_return", "Return"),
    ("episode_length_s", "Episode s"),
    ("terrain_level", "Terrain level"),
    ("velocity", "Velocity MSE"),
    ("adaboot_probability", "AdaBoot p"),
    ("value", "Value loss"),
    ("policy_kl", "KL"),
    ("learning_rate", "LR"),
)


def _format_iteration_log(iteration, total_iterations, metrics, eta_seconds):
    """iteration 번호, 요약 지표, 남은 시간을 터미널 출력 문자열로 만든다."""
    title = f" Learning iteration {iteration}/{total_iterations} "
    lines = ["#" * 80, title.center(80), ""]
    lines.extend(f"{label + ':':>35} {metrics[key]:.6g}" for key, label in _CONSOLE_FIELDS if key in metrics)
    lines.extend(["-" * 80, f"{'ETA:':>35} {eta_seconds:.1f}s"])
    return "\n".join(lines)


class OnPolicyRunner:
    """actor·critic·CENet을 하나의 PPO 학습 루프에서 함께 학습하고 평가하는 runner."""

    def __init__(self, env, train_cfg, run_io=None, device="cpu"):
        """환경 규격으로 모델·알고리즘·AdaBoot·rollout 저장소를 구성한다."""
        self._env = env
        self._cfg = copy.deepcopy(train_cfg)
        self._device = torch.device(device)
        self._run_io = run_io

        # 모델·알고리즘: DreamWaQ actor-critic 하나를 PPO와 AdaBoot로 학습한다.
        self._model = ActorCritic(env.num_obs, env.num_critic_obs, env.num_actions).to(self._device)
        log.info("model parameters=%d", sum(parameter.numel() for parameter in self._model.parameters()))
        self._algorithm = PPO(self._model)
        self._adaboot = AdaBoot(env.num_envs, self._device)

        # 학습 상태: rollout 저장소, 반복 횟수, 기록용 episode return·길이, 최고 지형 레벨을 준비한다.
        self._save_interval = self._cfg["runner"]["save_interval"]
        self._storage = RolloutStorage(env.num_envs, NUM_STEPS_PER_ENV, env.num_obs, env.num_critic_obs,
                                       env.num_obs * env.history_length, env.num_actions, self._device)
        self._iteration = 0
        self._episode_return = torch.zeros(env.num_envs, device=self._device)
        self._episode_steps = torch.zeros(env.num_envs, device=self._device)
        self._best_terrain_level = float("-inf")

    def learn(self, num_learning_iterations):
        """rollout 수집과 PPO 갱신을 반복하고 지표와 checkpoint를 기록한다."""
        # 학습 준비: 첫 학습이면 초기 관측으로 정규화 통계를 만들고 모델을 학습 모드로 둔다.
        observation = self._env.get_observations()
        if self._iteration == 0:
            self._model.update_normalizers(observation["obs"], observation["critic"])
        self._model.train()
        total_iterations = self._iteration + num_learning_iterations
        start_time = time.time()

        for local_iteration in range(num_learning_iterations):
            # rollout 수집: iteration 동안 AdaBoot 확률을 고정하고 num_steps만큼 전이를 저장한다.
            probability = self._adaboot.probability()
            completed_returns, completed_lengths = [], []
            log_sums, log_counts = {}, {}
            with torch.no_grad():
                for _ in range(NUM_STEPS_PER_ENV):
                    # 행동 샘플링: AdaBoot로 정책의 속도 입력을 고르고 행동·가치를 계산한다.
                    use_estimate = torch.rand(self._env.num_envs, device=self._device) < probability
                    policy = self._model.distribution(observation["obs"], observation["history"],
                                                      observation["velocity"], use_estimate)
                    actions = policy.sample()
                    values = self._model.evaluate(observation["critic"])
                    transition = {
                        "obs": observation["obs"], "history": observation["history"],
                        "critic": observation["critic"], "velocity": observation["velocity"],
                        "actions": actions, "use_estimate": use_estimate.float(),
                        "mean": policy.mean, "std": policy.scale,
                        "log_prob": policy.log_prob(actions).sum(-1), "values": values,
                    }

                    # 환경 진행: 시간 제한 종료는 현재 가치로 bootstrap한 보상을 저장한다.
                    observation, result = self._env.step(actions)
                    dones = result["dones"]
                    transition.update({
                        "rewards": result["rewards"] + GAMMA * values * result["time_outs"],
                        "dones": dones.float(), "next_obs": result["next_obs"],
                    })
                    self._storage.add(transition)
                    self._adaboot.update(result["rewards"], dones)

                    # 기록 누적: 완료 episode의 return·길이와 Isaac Lab manager 로그.
                    self._episode_return += result["rewards"]
                    self._episode_steps += 1
                    completed_returns.append(self._episode_return[dones].clone())
                    completed_lengths.append(self._episode_steps[dones].clone())
                    self._episode_return[dones] = 0.0
                    self._episode_steps[dones] = 0.0
                    for key, value in result["log"].items():
                        log_sums[key] = log_sums.get(key, 0.0) + torch.as_tensor(value, device=self._device)
                        log_counts[key] = log_counts.get(key, 0) + 1
                last_values = self._model.evaluate(observation["critic"])

            # 모델 갱신: PPO로 actor·critic·CENet을 갱신하고 사용한 rollout으로 정규화 통계를 갱신한다.
            self._storage.compute_returns(last_values, GAMMA, GAE_LAMBDA)
            metrics = self._algorithm.update(self._storage)
            self._model.update_normalizers(*self._storage.normalization_data())
            self._storage.clear()
            self._iteration += 1

            # 지표 기록: episode 통계·지형 레벨·AdaBoot 확률·manager 로그를 TensorBoard와 터미널에 기록한다.
            returns = torch.cat(completed_returns)
            lengths = torch.cat(completed_lengths)
            if returns.numel():
                metrics["episode_return"] = returns.mean().item()
                metrics["episode_length_s"] = (lengths.mean() * self._env.step_dt).item()
            terrain_level = self._env.terrain_level
            if terrain_level is not None:
                metrics["terrain_level"] = terrain_level.item()
            metrics["adaboot_probability"] = float(probability)
            metrics.update({f"Manager/{key}": (value / log_counts[key]).item() for key, value in log_sums.items()})
            if self._run_io is not None:
                self._run_io.log_metrics(self._iteration, metrics)
            elapsed = time.time() - start_time
            eta_seconds = elapsed / (local_iteration + 1) * (num_learning_iterations - local_iteration - 1)
            log.info(_format_iteration_log(self._iteration, total_iterations, metrics, eta_seconds))

            # checkpoint 저장: 저장 주기마다 저장한다.
            if self._run_io is not None and self._iteration % self._save_interval == 0:
                self._run_io.save_checkpoint(f"model_{self._iteration}.pt", self.state_dict())

            # best policy 저장: 평균 지형 레벨이 최고값을 넘으면 best checkpoint를 덮어쓴다.
            if "terrain_level" in metrics and metrics["terrain_level"] > self._best_terrain_level:
                self._best_terrain_level = metrics["terrain_level"]
                if self._run_io is not None:
                    self._run_io.save_checkpoint(BEST_CHECKPOINT_NAME, self.state_dict())
                    log.info("best policy saved: iteration=%d terrain_level=%.4f",
                             self._iteration, self._best_terrain_level)

        # 학습이 끝나면 마지막 checkpoint를 저장한다.
        if self._run_io is not None:
            self._run_io.save_checkpoint(f"model_{self._iteration}.pt", self.state_dict())

    def evaluate(self, num_steps):
        """결정적 정책으로 num_steps 진행하며 넘어짐·명령 추적 오차·속도 추정 오차를 집계한다."""
        env = self._env
        policy = self.get_inference_policy()
        observation = env.get_observations()
        falls = torch.zeros((), device=self._device)
        linear_error = torch.zeros((), device=self._device)
        yaw_error = torch.zeros((), device=self._device)
        yaw_bias = torch.zeros((), device=self._device)
        velocity_square_error = torch.zeros(3, device=self._device)

        # 평가 루프: 행동 전 추정 속도 오차를 누적하고 한 스텝씩 진행하며 넘어짐과 추적 오차를 누적한다.
        with torch.no_grad():
            for _ in range(num_steps):
                estimate = self._model.estimate_velocity(observation["history"])
                velocity_square_error += (estimate - observation["velocity"]).square().mean(0)
                observation, result = env.step(policy(observation["obs"], observation["history"]))
                command = result["command"]
                falls += result["terminated"].sum()
                linear_error += (observation["velocity"][:, :2] - command[:, :2]).norm(dim=-1).mean()
                yaw_difference = result["yaw_rate"] - command[:, 2]
                yaw_error += yaw_difference.abs().mean()
                yaw_bias += yaw_difference.mean()

        # 결과 집계: 스텝 평균 오차와 환경·분당 넘어짐 수를 정리한다.
        duration_min = num_steps * env.step_dt / 60.0
        return {
            "steps": num_steps, "num_envs": env.num_envs,
            "falls_total": int(falls.item()),
            "falls_per_env_minute": falls.item() / (env.num_envs * duration_min),
            "linear_error_mps": linear_error.item() / num_steps,
            "yaw_error_radps": yaw_error.item() / num_steps,
            "yaw_bias_radps": yaw_bias.item() / num_steps,
            "velocity_rmse_mps": (velocity_square_error / num_steps).sqrt().tolist(),
        }

    def state_dict(self):
        """checkpoint에 기록할 학습 상태를 반환한다."""
        return {
            "train_cfg": self._cfg,
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._algorithm.state_dict(),
            "adaboot": self._adaboot.state_dict(),
            "iteration": self._iteration,
            "best_terrain_level": self._best_terrain_level,
        }

    def load_state_dict(self, checkpoint, resume=True):
        """모델을 복원하고, resume이면 optimizer·AdaBoot·반복 횟수·최고 지형 레벨까지 복원한다."""
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._iteration = checkpoint["iteration"]
        if resume:
            self._algorithm.load_state_dict(checkpoint["optimizer_state_dict"])
            self._adaboot.load_state_dict(checkpoint["adaboot"])
            self._best_terrain_level = checkpoint.get("best_terrain_level", float("-inf"))
        log.info("checkpoint loaded: iteration=%d resume=%s", self._iteration, resume)

    def get_inference_policy(self):
        """평가 모드로 전환한 결정적 정책을 반환한다."""
        self._model.eval()
        return self._model.act_inference
