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


import torch
from torch import nn


class PPO:
    """저장된 rollout으로 clipped surrogate·가치·엔트로피·추정기 손실을 최적화한다."""

    def __init__(self, actor_critic, learning_rate=1e-3, num_learning_epochs=5,
                 num_mini_batches=4, clip_param=0.2, gamma=0.99, lam=0.95,
                 value_loss_coef=1.0, entropy_coef=0.01, max_grad_norm=1.0,
                 use_clipped_value_loss=True, schedule="adaptive", desired_kl=0.01,
                 velocity_loss_coef=1.0, prediction_loss_coef=1.0, beta=1.0):
        """최적화 대상 모델, 옵티마이저, 손실 가중치를 저장한다."""
        # 정책·가치·추정기 파라미터 전체를 하나의 Adam으로 최적화한다.
        self._actor_critic = actor_critic
        self._optimizer = torch.optim.Adam(actor_critic.parameters(), lr=learning_rate)

        # 학습 설정: 반복 횟수, GAE 계수, PPO·추정기 손실 가중치, 학습률 조정 방식을 저장한다.
        self._epochs = num_learning_epochs
        self._batches = num_mini_batches
        self._clip = clip_param
        self._gamma, self._lam = gamma, lam
        self._value_coef, self._entropy_coef = value_loss_coef, entropy_coef
        self._grad_norm = max_grad_norm
        self._clipped_value = use_clipped_value_loss
        self._schedule, self._desired_kl = schedule, desired_kl
        self._velocity_coef, self._prediction_coef = velocity_loss_coef, prediction_loss_coef
        self._beta = beta

    @property
    def gamma(self):
        """GAE 계산에 사용할 할인율을 반환한다."""
        return self._gamma

    @property
    def lam(self):
        """GAE 계산에 사용할 λ 계수를 반환한다."""
        return self._lam

    def update(self, storage):
        """rollout 전체를 epoch·미니배치 단위로 반복 학습하고 손실 평균을 반환한다."""
        totals = {}
        count = 0
        for batch in storage.batches(self._batches, self._epochs):
            # 모델 출력: rollout과 같은 입력 조건으로 현재 정책 분포·가치·추정기 손실을 계산한다.
            policy, velocity, prediction, latent_kl = self._actor_critic.training_terms(
                batch["obs"], batch["history"], batch["velocity"], batch["use_estimate"],
                batch["next_obs"], 1.0 - batch["dones"])
            log_prob = policy.log_prob(batch["actions"]).sum(-1)
            values = self._actor_critic.evaluate(batch["critic"])

            # 학습률 조정: rollout 정책과 현재 정책의 KL이 목표 범위를 벗어나면 학습률을 바꾼다.
            with torch.no_grad():
                kl = (torch.log(policy.scale / batch["std"])
                      + (batch["std"].square() + (batch["mean"] - policy.mean).square())
                      / (2 * policy.scale.square()) - 0.5).sum(-1).mean()
                if self._schedule == "adaptive":
                    rate = self._optimizer.param_groups[0]["lr"]
                    if kl > 2 * self._desired_kl:
                        rate = max(1e-5, rate / 1.5)
                    elif 0 < kl < self._desired_kl / 2:
                        rate = min(1e-2, rate * 1.5)
                    for group in self._optimizer.param_groups:
                        group["lr"] = rate

            # 전체 손실: clipped surrogate·clipped 가치·엔트로피 PPO 손실에 추정기 손실을 더한다.
            ratio = (log_prob - batch["log_prob"]).exp()
            surrogate = torch.maximum(
                -batch["advantages"] * ratio,
                -batch["advantages"] * ratio.clamp(1-self._clip, 1+self._clip)).mean()
            value_error = (values - batch["returns"]).square()
            if self._clipped_value:
                clipped = batch["values"] + (values - batch["values"]).clamp(-self._clip, self._clip)
                value_error = torch.maximum(value_error, (clipped - batch["returns"]).square())
            value_loss = value_error.mean()
            entropy = policy.entropy().sum(-1).mean()
            loss = (surrogate + self._value_coef * value_loss - self._entropy_coef * entropy
                    + self._velocity_coef * velocity + self._prediction_coef * prediction
                    + self._beta * latent_kl)

            # 파라미터 갱신: gradient norm을 제한해 한 번 갱신하고 지표를 GPU 텐서로 누적한다.
            self._optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self._actor_critic.parameters(), self._grad_norm)
            self._optimizer.step()
            metrics = {"surrogate": surrogate, "value": value_loss, "velocity": velocity,
                       "prediction": prediction, "latent_kl": latent_kl, "policy_kl": kl,
                       "entropy": entropy, "gradient_norm": grad_norm}
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value.detach()
            count += 1

        # 누적한 지표를 업데이트 횟수로 평균해 한 번에 CPU로 옮긴다.
        result = {key: (value / count).item() for key, value in totals.items()}
        result["learning_rate"] = self._optimizer.param_groups[0]["lr"]
        return result

    def state_dict(self):
        """학습 재개에 필요한 옵티마이저 상태를 반환한다."""
        return self._optimizer.state_dict()

    def load_state_dict(self, state):
        """저장된 옵티마이저 상태를 복원한다."""
        self._optimizer.load_state_dict(state)
