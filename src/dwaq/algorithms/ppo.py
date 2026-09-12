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


"""DreamWaQ 보조 손실을 함께 최적화하는 PPO."""
import torch
from torch import nn


class PPO:
    """고정된 정규화와 AdaBoot 선택으로 PPO ratio를 계산한다."""

    def __init__(self, actor_critic, learning_rate=1e-3, num_learning_epochs=5,
                 num_mini_batches=4, clip_param=0.2, gamma=0.99, lam=0.95,
                 value_loss_coef=1.0, entropy_coef=0.01, max_grad_norm=1.0,
                 use_clipped_value_loss=True, schedule="adaptive", desired_kl=0.01,
                 velocity_loss_coef=1.0, prediction_loss_coef=1.0, beta=1.0):
        """정책과 보조 손실의 가중치를 명시적으로 저장한다."""
        self.actor_critic = actor_critic
        self._optimizer = torch.optim.Adam(actor_critic.parameters(), lr=learning_rate)
        self._epochs = num_learning_epochs
        self._batches = num_mini_batches
        self._clip = clip_param
        self.gamma, self.lam = gamma, lam
        self._value_coef, self._entropy_coef = value_loss_coef, entropy_coef
        self._grad_norm = max_grad_norm
        self._clipped_value = use_clipped_value_loss
        self._schedule, self._desired_kl = schedule, desired_kl
        self._velocity_coef, self._prediction_coef = velocity_loss_coef, prediction_loss_coef
        self._beta = beta

    def update(self, storage):
        """각 손실을 표본 평균으로 계산하고 학습 결과를 반환한다."""
        totals = {}
        count = 0
        for batch in storage.batches(self._batches, self._epochs):
            policy = self.actor_critic.distribution(
                batch["obs"], batch["history"], batch["velocity"], batch["use_estimate"])
            log_prob = policy.log_prob(batch["actions"]).sum(-1)
            values = self.actor_critic.evaluate(batch["critic"])
            with torch.no_grad():
                kl = (torch.log(policy.scale / batch["std"])
                      + (batch["std"].square() + (batch["mean"] - policy.mean).square())
                      / (2 * policy.scale.square()) - 0.5).sum(-1).mean()
                if self._schedule == "adaptive" and self._desired_kl is not None:
                    rate = self._optimizer.param_groups[0]["lr"]
                    if kl > 2 * self._desired_kl:
                        rate = max(1e-5, rate / 1.5)
                    elif 0 < kl < self._desired_kl / 2:
                        rate = min(1e-2, rate * 1.5)
                    for group in self._optimizer.param_groups:
                        group["lr"] = rate
            ratio = (log_prob - batch["log_prob"]).exp()
            surrogate = torch.maximum(
                -batch["advantages"] * ratio,
                -batch["advantages"] * ratio.clamp(1-self._clip, 1+self._clip)).mean()
            value_error = (values - batch["returns"]).square()
            if self._clipped_value:
                clipped = batch["values"] + (values - batch["values"]).clamp(-self._clip, self._clip)
                value_error = torch.maximum(value_error, (clipped - batch["returns"]).square())
            value_loss = value_error.mean()
            velocity, prediction, latent_kl = self.actor_critic.auxiliary_losses(
                batch["history"], batch["velocity"], batch["next_obs"], batch["prediction_valid"])
            entropy = policy.entropy().sum(-1).mean()
            loss = (surrogate + self._value_coef * value_loss - self._entropy_coef * entropy
                    + self._velocity_coef * velocity + self._prediction_coef * prediction
                    + self._beta * latent_kl)
            if not torch.isfinite(loss):
                raise FloatingPointError("PPO/CENet 손실에 비유한 값이 있습니다.")
            self._optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self._grad_norm,
                                                error_if_nonfinite=True)
            self._optimizer.step()
            metrics = {"surrogate": surrogate, "value": value_loss, "velocity": velocity,
                       "prediction": prediction, "latent_kl": latent_kl, "policy_kl": kl,
                       "entropy": entropy, "gradient_norm": grad_norm}
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value.detach()
            count += 1
        result = {key: (value / count).item() for key, value in totals.items()}
        result["learning_rate"] = self._optimizer.param_groups[0]["lr"]
        return result

    def state_dict(self):
        """optimizer 재개 상태를 반환한다."""
        return self._optimizer.state_dict()

    def load_state_dict(self, state):
        """optimizer 상태를 복원한다."""
        self._optimizer.load_state_dict(state)
