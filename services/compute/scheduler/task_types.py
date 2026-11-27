
"""Типы и контракты вычислительных задач LAIE."""

from dataclasses import dataclass
from enum import Enum


class TaskType(str, Enum):
    """Поддерживаемые категории вычислительных задач."""

    MATRIX_MULTIPLICATION = "matrix_multiplication"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    COMBINATION_GENERATION = "combination_generation"
    MONTE_CARLO = "monte_carlo"
    GENETIC_OPTIMIZATION = "genetic_optimization"
    MODEL_TRAINING = "model_training"
    HYPOTHESIS_TESTING = "hypothesis_testing"


class WorkloadKind(str, Enum):
    """Характер вычислительной нагрузки."""

    CPU_BOUND = "cpu_bound"
    GPU_PARALLEL = "gpu_parallel"
    MIXED = "mixed"


class PrecisionRequirement(str, Enum):
    """Требования к точности результата."""

    EXACT = "exact"
    INTEGER = "integer"
    FLOAT32 = "float32"
    FLOAT64 = "float64"


class ComputeBackend(str, Enum):
    """Допустимые вычислительные устройства."""

    CPU = "cpu"
    CUDA = "cuda"


@dataclass(frozen=True, slots=True)
class TaskDefinition:
    """Неизменяемое описание вычислительной задачи."""

    task_type: TaskType
    name: str
    description: str
    workload: WorkloadKind
    supported_backends: tuple[ComputeBackend, ...]
    precision: PrecisionRequirement
    parallelizable: bool
    supports_batching: bool
    estimator_key: str

    def __post_init__(self) -> None:
        """Проверяет корректность описания задачи."""
        if not isinstance(self.task_type, TaskType):
            raise TypeError("task_type должен быть элементом TaskType")

        if not isinstance(self.workload, WorkloadKind):
            raise TypeError("workload должен быть элементом WorkloadKind")

        if not isinstance(self.precision, PrecisionRequirement):
            raise TypeError(
                "precision должен быть элементом PrecisionRequirement"
            )

        if not self.name.strip():
            raise ValueError("Название задачи не может быть пустым")

        if not self.description.strip():
            raise ValueError("Описание задачи не может быть пустым")

        if not self.estimator_key.strip():
            raise ValueError("estimator_key не может быть пустым")

        if not self.supported_backends:
            raise ValueError(
                "Необходимо указать хотя бы одно вычислительное устройство"
            )

        if len(set(self.supported_backends)) != len(
            self.supported_backends
        ):
            raise ValueError("Список вычислительных устройств содержит дубли")

        if any(
            not isinstance(backend, ComputeBackend)
            for backend in self.supported_backends
        ):
            raise TypeError(
                "supported_backends должен содержать ComputeBackend"
            )

        if not isinstance(self.parallelizable, bool):
            raise TypeError("parallelizable должен иметь тип bool")

        if not isinstance(self.supports_batching, bool):
            raise TypeError("supports_batching должен иметь тип bool")

    def supports_backend(self, backend: ComputeBackend) -> bool:
        """Проверяет поддержку указанного устройства."""
        if not isinstance(self.supported_backends, tuple):
            raise TypeError("supported_backends должен быть tuple")

        if not self.supported_backends:
            raise ValueError(
                "Необходимо указать хотя бы одно вычислительное устройство"
            )

        if any(
            not isinstance(backend, ComputeBackend)
            for backend in self.supported_backends
        ):
            raise TypeError(
                "supported_backends должен содержать ComputeBackend"
            )

        if len(set(self.supported_backends)) != len(
            self.supported_backends
        ):
            raise ValueError("Список вычислительных устройств содержит дубли")
        return backend in self.supported_backends
