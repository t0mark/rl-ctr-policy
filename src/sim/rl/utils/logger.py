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

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process


class Logger:
    """롤아웃 중 로봇 상태/보상을 기록하고, 별도 프로세스에서 그래프로 시각화하는 로거."""

    def __init__(self, dt: float, num_dof: int):
        """
        Args:
            dt (float): 한 스텝의 시간 간격(초)
            num_dof (int): 로봇의 전체 관절 수. 관절별 그래프 개수/배치를 결정하는 데 쓰이며,
                이 값을 통해 로봇이 바뀌어도(예: 12->15) 코드 수정 없이 그래프가 자동으로 늘어난다.
        """
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_dof = num_dof
        self.num_episodes = 0
        self.plot_process = None

    def log_state(self, key, value):
        """상태 값 하나를 기록한다."""
        self.state_log[key].append(value)

    def log_states(self, dict):
        """딕셔너리에 담긴 여러 상태 값을 한 번에 기록한다."""
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        """에피소드 보상 항목을 기록한다."""
        for key, value in dict.items():
            if 'rew' in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        """기록된 로그를 모두 비운다."""
        self.state_log.clear()
        self.rew_log.clear()

    def plot_states(self):
        """학습/평가 루프를 막지 않도록 별도 프로세스에서 시각화를 실행한다."""
        self.plot_process = Process(target=self._plot)
        self.plot_process.start()

    def _plot(self):
        """
        고정된 요약 그래프(속도 추종, 접촉력, 토크 등) 9칸 + 관절별 위치 그래프를 그린다.
        관절별 그래프는 self.num_dof 개수만큼 동적으로 배치되므로, 로봇의 관절 수가
        바뀌어도(예: 사족보행 12 -> G1 15/29) 그래프 배치 코드를 고칠 필요가 없다.
        """
        num_summary_rows = 3
        num_cols = 3
        num_joint_rows = int(np.ceil(self.num_dof / num_cols))
        nb_rows = num_summary_rows + num_joint_rows

        fig, axs = plt.subplots(nb_rows, num_cols)
        log = self.state_log
        for key, value in log.items():
            time = np.linspace(0, len(value) * self.dt, len(value))
            break

        # ---- 요약 그래프 (관절 수와 무관하게 항상 같은 자리) ----
        a = axs[1, 0]
        if log["dof_pos"]: a.plot(time, log["dof_pos"], label='measured')
        if log["dof_pos_target"]: a.plot(time, log["dof_pos_target"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position')
        a.legend()

        a = axs[1, 1]
        if log["dof_vel"]: a.plot(time, log["dof_vel"], label='measured')
        if log["dof_vel_target"]: a.plot(time, log["dof_vel_target"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity')
        a.legend()

        a = axs[0, 0]
        if log["base_vel_x"]: a.plot(time, log["base_vel_x"], label='measured')
        if log["command_x"]: a.plot(time, log["command_x"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity x')
        a.legend()

        a = axs[0, 1]
        if log["base_vel_y"]: a.plot(time, log["base_vel_y"], label='measured')
        if log["command_y"]: a.plot(time, log["command_y"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity y')
        a.legend()

        a = axs[0, 2]
        if log["base_vel_yaw"]: a.plot(time, log["base_vel_yaw"], label='measured')
        if log["command_yaw"]: a.plot(time, log["command_yaw"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base ang vel [rad/s]', title='Base velocity yaw')
        a.legend()

        a = axs[1, 2]
        if log["base_vel_z"]: a.plot(time, log["base_vel_z"], label='measured')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity z')
        a.legend()

        a = axs[2, 0]
        if log["contact_forces_z"]:
            forces = np.array(log["contact_forces_z"])
            for i in range(forces.shape[1]):
                a.plot(time, forces[:, i], label=f'force {i}')
        a.set(xlabel='time [s]', ylabel='Forces z [N]', title='Vertical Contact forces')
        a.legend()

        a = axs[2, 1]
        if log["dof_vel"] != [] and log["dof_torque"] != []: a.plot(log["dof_vel"], log["dof_torque"], 'x', label='measured')
        a.set(xlabel='Joint vel [rad/s]', ylabel='Joint Torque [Nm]', title='Torque/velocity curves')
        a.legend()

        a = axs[2, 2]
        if log["dof_torque"] != []: a.plot(time, log["dof_torque"], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='Torque')
        a.legend()

        # ---- 관절별 위치 그래프: self.num_dof 개수만큼 동적으로 생성 ----
        for i in range(self.num_dof):
            row = num_summary_rows + i // num_cols
            col = i % num_cols
            a = axs[row, col]
            key = f"dof_pos_{i}"
            if log[key]:
                a.plot(time, log[key], label='measured')
            a.set(xlabel='time [s]', ylabel='Position [rad]', title=f'joint {i} Position')
            a.legend()

        plt.show()

    def print_rewards(self):
        """에피소드당 평균 보상을 출력한다."""
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")

    def __del__(self):
        """백그라운드 시각화 프로세스가 남아있다면 정리한다."""
        if self.plot_process is not None:
            self.plot_process.kill()
