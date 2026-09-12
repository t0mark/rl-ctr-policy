"""PD 토크에 독립적인 motor strength 배율을 적용한다."""
import torch
from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg
from isaaclab.utils import configclass


class StrengthPDActuator(IdealPDActuator):
    """배율을 적용한 PD 토크를 고정된 effort 한계로 제한한다."""

    def __init__(self, *args, **kwargs):
        """환경·관절별 출력 배율을 생성한다."""
        super().__init__(*args, **kwargs)
        self._strength = torch.ones_like(self.stiffness)

    def randomize_strength(self, env_ids, factor_range):
        """선택한 환경의 배율을 독립적으로 샘플링한다."""
        ids = slice(None) if env_ids is None else env_ids
        self._strength[ids] = torch.empty_like(self._strength[ids]).uniform_(*factor_range)

    def _clip_effort(self, effort):
        """PD gain을 보존하고 출력 토크에 배율과 포화를 적용한다."""
        return super()._clip_effort(effort * self._strength)


@configclass
class StrengthPDActuatorCfg(IdealPDActuatorCfg):
    """physics tick에서 직접 토크를 계산하는 actuator 설정."""

    class_type: type = StrengthPDActuator
