"""Head modules for downstream tasks."""

from src.model.heads.base_head import BaseHead, TaskType
from src.model.heads.rnn_head import RNNHead

__all__ = ["BaseHead", "TaskType", "RNNHead"]

