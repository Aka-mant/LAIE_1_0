
"""Типы данных и статусы выполнения вычислительных задач."""

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Any

from .task_types import ComputeBackend, TaskType


class ExecutionStatus(StrEnum):
    """Статус выполнения задачи."""

    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExecutionErrorCode(StrEnum):
    """Коды ошибок исполнения."""

    NONE = "none"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    MEMORY_ERROR = "memory_error"
    EXECUTION_ERROR = "execution_error"
    VALIDATION_ERROR = "validation_error"
    CANCELLED = "cancelled"
    INVALID_RESULT = "invalid_result"


@dataclass(frozen=True, slots=True)
class ExecutionMetrics:
    """Фактические метрики выполнения задачи."""

    execution_seconds: float
    preparation_seconds: float
    validation_seconds: float
    batch_count: int
    processed_items: int
    peak_memory_bytes: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "execution_seconds",
            "preparation_seconds",
            "validation_seconds",
        ):
            value = getattr(self, name)

            if isinstance(value, bool) or not isinstance(
                value,
                (int, float),
            ):
                raise TypeError(f"{name} должен быть числом.")

            if not isfinite(value) or value < 0:
                raise ValueError(
                    f"{name} должен быть конечным неотрицательным числом."
                )

        for name in ("batch_count", "processed_items"):
            value = getattr(self, name)

            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} должен быть целым числом.")

            if value < 0:
                raise ValueError(f"{name} не может быть отрицательным.")

        if self.peak_memory_bytes is not None:
            if isinstance(self.peak_memory_bytes, bool) or not isinstance(
                self.peak_memory_bytes,
                int,
            ):
                raise TypeError("peak_memory_bytes должен быть целым числом.")

            if self.peak_memory_bytes < 0:
                raise ValueError(
                    "peak_memory_bytes не может быть отрицательным."
                )


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Результат выполнения задачи."""

    task_type: TaskType
    status: ExecutionStatus
    requested_backend: ComputeBackend
    actual_backend: ComputeBackend | None
    result: Any
    metrics: ExecutionMetrics
    error_code: ExecutionErrorCode = ExecutionErrorCode.NONE
    error_message: str = ""
    fallback_used: bool = False
    result_validated: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.task_type, TaskType):
            raise TypeError("task_type должен быть TaskType.")

        if not isinstance(self.status, ExecutionStatus):
            raise TypeError("status должен быть ExecutionStatus.")

        if not isinstance(self.requested_backend, ComputeBackend):
            raise TypeError(
                "requested_backend должен быть ComputeBackend."
            )

        if self.actual_backend is not None and not isinstance(
            self.actual_backend,
            ComputeBackend,
        ):
            raise TypeError("actual_backend должен быть ComputeBackend.")

        if not isinstance(self.metrics, ExecutionMetrics):
            raise TypeError("metrics должен быть ExecutionMetrics.")

        if not isinstance(self.error_code, ExecutionErrorCode):
            raise TypeError("error_code должен быть ExecutionErrorCode.")

        if not isinstance(self.error_message, str):
            raise TypeError("error_message должен быть строкой.")

        if not isinstance(self.fallback_used, bool):
            raise TypeError("fallback_used должен быть bool.")

        if not isinstance(self.result_validated, bool):
            raise TypeError("result_validated должен быть bool.")

        if self.status == ExecutionStatus.SUCCESS:
            if self.actual_backend is None:
                raise ValueError(
                    "Успешное выполнение должно иметь actual_backend."
                )

            if self.error_code != ExecutionErrorCode.NONE:
                raise ValueError(
                    "Успешное выполнение не может содержать код ошибки."
                )

            if not self.result_validated:
                raise ValueError(
                    "Успешное выполнение должно быть проверено."
                )

        if self.status == ExecutionStatus.FAILED:
            if self.error_code == ExecutionErrorCode.NONE:
                raise ValueError(
                    "Неуспешное выполнение должно иметь код ошибки."
                )

        if self.status == ExecutionStatus.CANCELLED:
            if self.error_code != ExecutionErrorCode.CANCELLED:
                raise ValueError(
                    "Отменённое выполнение должно иметь код CANCELLED."
                )

    def as_dict(self) -> dict[str, Any]:
        """Возвращает JSON-совместимое представление результата."""
        return {
            "task_type": self.task_type.value,
            "status": self.status.value,
            "requested_backend": self.requested_backend.value,
            "actual_backend": (
                self.actual_backend.value
                if self.actual_backend is not None
                else None
            ),
            "metrics": {
                "execution_seconds": self.metrics.execution_seconds,
                "preparation_seconds": self.metrics.preparation_seconds,
                "validation_seconds": self.metrics.validation_seconds,
                "batch_count": self.metrics.batch_count,
                "processed_items": self.metrics.processed_items,
                "peak_memory_bytes": self.metrics.peak_memory_bytes,
            },
            "error_code": self.error_code.value,
            "error_message": self.error_message,
            "fallback_used": self.fallback_used,
            "result_validated": self.result_validated,
        }
