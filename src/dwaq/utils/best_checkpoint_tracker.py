# Copyright (c) 2026 DreamWaQ Project
# SPDX-License-Identifier: BSD-3-Clause

"""명령 curriculum 범위가 최대치에 도달한 뒤의 episode return으로 best 체크포인트 갱신 여부를 판정한다."""

import logging
import math
from collections import deque

log = logging.getLogger(__name__)

# episode return 이동평균에 사용하는 iteration 수.
RETURN_AVERAGE_WINDOW = 20


class BestCheckpointTracker:
    """선속도 명령 범위가 설정 최대치에 도달한 iteration의 episode return 이동평균 최고값을 추적한다.

    범위가 최대치가 아닌 iteration에서는 이동평균 창을 비우고 판정하지 않으며,
    창이 가득 찬 뒤 이동평균이 이전 최고값을 넘으면 갱신으로 판정한다.
    """

    def __init__(self):
        """return 이동평균 창과 최고값을 초기화한다."""
        self._recent_returns = deque(maxlen=RETURN_AVERAGE_WINDOW)
        self._best_return_average = -math.inf
        self._range_complete = False

    @property
    def best_return_average(self):
        """지금까지 판정한 episode return 이동평균의 최고값을 반환한다."""
        return self._best_return_average

    def update(self, episode_return, curriculum_state):
        """이번 iteration의 episode return을 반영하고 best 갱신 여부를 반환한다.

        episode_return은 완료 episode가 없으면 None이고, curriculum_state는 명령 curriculum이 없으면 None이다.
        """
        # 선속도 명령 범위가 최대치가 아니면 이동평균 창을 비우고 판정하지 않는다.
        range_complete = self._is_linear_range_complete(curriculum_state)
        if range_complete != self._range_complete:
            log.info("best 체크포인트 판정 %s: 선속도 명령 범위 최대치 도달=%s",
                     "시작" if range_complete else "중단", range_complete)
            self._range_complete = range_complete
        if not range_complete:
            self._recent_returns.clear()
            return False

        # 완료 episode가 있는 iteration의 return만 창에 넣고, 창이 가득 차야 판정한다.
        if episode_return is None:
            return False
        self._recent_returns.append(episode_return)
        if len(self._recent_returns) < RETURN_AVERAGE_WINDOW:
            return False

        # 이동평균이 이전 최고값을 넘으면 최고값을 갱신한다.
        return_average = sum(self._recent_returns) / RETURN_AVERAGE_WINDOW
        if return_average <= self._best_return_average:
            return False
        self._best_return_average = return_average
        return True

    @staticmethod
    def _is_linear_range_complete(curriculum_state):
        """x·y 선속도 명령의 현재 범위가 설정 최대 범위와 같은지 확인한다."""
        if curriculum_state is None:
            return True
        low_reached = curriculum_state["low"][:2] <= curriculum_state["limit_low"][:2]
        high_reached = curriculum_state["high"][:2] >= curriculum_state["limit_high"][:2]
        return bool((low_reached & high_reached).all())
