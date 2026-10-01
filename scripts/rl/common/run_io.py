"""학습 지표와 checkpoint 파일의 입출력을 한곳에서 관리한다."""

from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter


class RunIO:
    """한 실행 폴더의 TensorBoard 기록과 checkpoint 저장을 관리한다."""

    def __init__(self, run_dir=None):
        """실행 폴더가 있으면 TensorBoard writer를 생성한다."""
        self._run_dir = Path(run_dir) if run_dir is not None else None
        self._writer = SummaryWriter(str(self._run_dir)) if self._run_dir is not None else None

    def log_metrics(self, iteration, metrics):
        """학습 지표를 TensorBoard에 기록한다."""
        if self._writer is None:
            return
        for key, value in metrics.items():
            self._writer.add_scalar(key, value, iteration)

    def save_checkpoint(self, filename, state):
        """실행 폴더에 checkpoint를 저장한다."""
        if self._run_dir is None:
            return
        torch.save(state, self._run_dir / filename)

    @staticmethod
    def load_checkpoint(path, device="cpu"):
        """checkpoint 파일을 지정한 장치로 읽는다."""
        return torch.load(path, map_location=device, weights_only=True)

    def close(self):
        """TensorBoard writer를 닫는다."""
        if self._writer is not None:
            self._writer.close()

