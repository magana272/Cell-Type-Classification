from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from typing import Any

import numpy as np
import optuna


@dataclass
class BaseHParams:
    lr: float
    weight_decay: float
    dropout: float
    label_smoothing: float
    optimizer: str
    loss: str
    focal_gamma: float
    normalize: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MLPHParams(BaseHParams):
    n_layers: int = 1
    hidden_dim: int = 256


@dataclass
class CNNHParams(BaseHParams):
    n_stages: int = 4


@dataclass
class GNNHParams(BaseHParams):
    n_layers: int = 1
    hidden_dim: int = 128
    k_neighbors: int = 7


@dataclass
class TransformerHParams(BaseHParams):
    n_layers: int = 1
    n_heads: int = 4
    embed_dim: int = 48


@dataclass
class BaseModelKwargs:
    dropout: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MLPModelKwargs(BaseModelKwargs):
    n_layers: int = 1
    hidden_dim: int = 256


@dataclass
class CNNModelKwargs(BaseModelKwargs):
    n_stages: int = 4
    use_checkpointing: bool = True


@dataclass
class GNNModelKwargs(BaseModelKwargs):
    n_layers: int = 1
    hidden_dim: int = 128


@dataclass
class TransformerModelKwargs(BaseModelKwargs):
    n_layers: int = 1
    n_heads: int = 4
    embed_dim: int = 48


class TrainConfig(ABC):
    @abstractmethod
    def suggest_hparams(self, trial: optuna.trial.Trial) -> BaseHParams:
        ...

    @abstractmethod
    def model_kwargs_from_params(self, params: BaseHParams) -> BaseModelKwargs:
        ...

    @abstractmethod
    def infer_model_kwargs(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        ...


@dataclass
class ExperimentConfig:
    model: str
    seed: int = 42
    batch_size: int = 8192
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-6
    optimizer: str = 'adamw'
    loss: str = 'cross_entropy'
    label_smoothing: float = 0.1
    focal_gamma: float = 2.0
    dropout: float = 0.1
    normalize: str | None = None
    n_hvg: int = 0
    k_neighbors: int = 15
    accumulation_steps: int = 1
    device: str = 'auto'

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)


@dataclass
class EvalMetrics:
    accuracy: float
    f1_macro: float
    f1_weighted: float
    precision_macro: float
    precision_weighted: float
    recall_macro: float
    recall_weighted: float
    confusion_matrix: np.ndarray | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.confusion_matrix is not None:
            d['confusion_matrix'] = self.confusion_matrix
        return d

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)


@dataclass
class ModelPredictions:
    y_true: np.ndarray
    y_pred: np.ndarray
    y_probs: np.ndarray
    class_names: list[str]
    n_classes: int

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)
