
"""Типизированные модели эксперимента точности прогнозирования LAIE."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Mapping

from services.compute.scheduler.task_types import ComputeBackend


class ExperimentSplit(str, Enum):
    """Разделение конфигураций эксперимента."""

    CALIBRATION = "calibration"
    VALIDATION = "validation"


class ExperimentStatus(str, Enum):
    """Состояние эксперимента."""

    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class MeasurementStatus(str, Enum):
    """Результат отдельного измерения."""

    SUCCESS = "success"
    SKIPPED = "skipped"
    FAILED = "failed"
    FALLBACK = "fallback"


class DecisionQuality(str, Enum):
    """Качество решения планировщика по фактическим измерениям."""

    CORRECT = "correct"
    SUBOPTIMAL = "suboptimal"
    INDETERMINATE = "indeterminate"
    NOT_COMPARABLE = "not_comparable"


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Настройки эксперимента точности прогнозирования."""

    calibration_sizes: tuple[int, ...] = (256, 512, 1024)
    validation_sizes: tuple[int, ...] = (768, 1536, 2048)
    batch_sizes: tuple[int, ...] = (64, 128, 256)
    repeats: int = 10
    cuda_warmup: int = 3
    random_seed: int = 42
    dtype: str = "float64"
    minimum_gpu_speedup: float = 0.10
    high_confidence_samples: int = 5

    def __post_init__(self) -> None:
        """Проверяет корректность конфигурации."""
        for name in (
            "calibration_sizes",
            "validation_sizes",
            "batch_sizes",
        ):
            values = getattr(self, name)

            if not isinstance(values, tuple):
                raise TypeError(f"{name} должен быть tuple")

            if not values:
                raise ValueError(f"{name} не может быть пустым")

            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in values
            ):
                raise TypeError(
                    f"{name} должен содержать только целые числа"
                )

            if any(value <= 0 for value in values):
                raise ValueError(
                    f"{name} должен содержать только положительные числа"
                )

            if len(set(values)) != len(values):
                raise ValueError(f"{name} содержит повторяющиеся значения")

        overlap = set(self.calibration_sizes) & set(self.validation_sizes)
        if overlap:
            raise ValueError(
                "Размеры калибровки и проверки должны быть независимы: "
                f"пересечение {sorted(overlap)}"
            )

        for name in (
            "repeats",
            "cuda_warmup",
            "random_seed",
            "high_confidence_samples",
        ):
            value = getattr(self, name)

            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} должен быть целым числом")

        if self.repeats < 2:
            raise ValueError("repeats должен быть не меньше 2")

        if self.cuda_warmup < 0:
            raise ValueError("cuda_warmup не может быть отрицательным")

        if self.high_confidence_samples < 2:
            raise ValueError(
                "high_confidence_samples должен быть не меньше 2"
            )

        if not isinstance(self.dtype, str) or self.dtype != "float64":
            raise ValueError(
                "В первой версии эксперимента поддерживается только float64"
            )

        if (
            isinstance(self.minimum_gpu_speedup, bool)
            or not isinstance(self.minimum_gpu_speedup, (int, float))
        ):
            raise TypeError("minimum_gpu_speedup должен быть числом")

        if (
            not math.isfinite(self.minimum_gpu_speedup)
            or not 0 <= self.minimum_gpu_speedup < 1
        ):
            raise ValueError(
                "minimum_gpu_speedup должен находиться в диапазоне [0, 1)"
            )


@dataclass(frozen=True, slots=True)
class ExperimentCase:
    """Одна уникальная конфигурация матричной задачи."""

    split: ExperimentSplit
    matrix_m: int
    matrix_k: int
    matrix_n: int
    batch_size: int
    dtype: str
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.split, ExperimentSplit):
            raise TypeError("split должен быть ExperimentSplit")

        for name in (
            "matrix_m",
            "matrix_k",
            "matrix_n",
            "batch_size",
            "seed",
        ):
            value = getattr(self, name)

            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} должен быть целым числом")

        for name in ("matrix_m", "matrix_k", "matrix_n", "batch_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} должен быть положительным")

        if not isinstance(self.dtype, str) or self.dtype != "float64":
            raise ValueError("ExperimentCase поддерживает только float64")

    @property
    def case_id(self) -> str:
        """Возвращает стабильный идентификатор конфигурации."""
        return (
            f"{self.split.value}:"
            f"{self.matrix_m}x{self.matrix_k}x{self.matrix_n}:"
            f"batch={self.batch_size}:"
            f"dtype={self.dtype}:"
            f"seed={self.seed}"
        )

    @property
    def operations(self) -> int:
        """Количество операций матричного умножения."""
        return 2 * self.matrix_m * self.matrix_k * self.matrix_n

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "split": self.split.value,
            "matrix_m": self.matrix_m,
            "matrix_k": self.matrix_k,
            "matrix_n": self.matrix_n,
            "batch_size": self.batch_size,
            "dtype": self.dtype,
            "seed": self.seed,
            "operations": self.operations,
        }


@dataclass(frozen=True, slots=True)
class ExperimentMeasurement:
    """Одно фактическое измерение выполнения матричной задачи."""

    case_id: str
    requested_backend: ComputeBackend
    actual_backend: ComputeBackend | None
    status: MeasurementStatus
    repeat_index: int
    operations: int
    input_bytes: int
    output_bytes: int
    compute_seconds: float | None = None
    transfer_seconds: float | None = None
    total_seconds: float | None = None
    peak_memory_bytes: int | None = None
    fallback_used: bool = False
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError("case_id не может быть пустым")

        if not isinstance(self.requested_backend, ComputeBackend):
            raise TypeError("requested_backend должен быть ComputeBackend")

        if (
            self.actual_backend is not None
            and not isinstance(self.actual_backend, ComputeBackend)
        ):
            raise TypeError("actual_backend должен быть ComputeBackend или None")

        if not isinstance(self.status, MeasurementStatus):
            raise TypeError("status должен быть MeasurementStatus")

        for name in (
            "repeat_index",
            "operations",
            "input_bytes",
            "output_bytes",
        ):
            value = getattr(self, name)

            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} должен быть целым числом")

            if value < 0:
                raise ValueError(f"{name} не может быть отрицательным")

        if self.operations == 0:
            raise ValueError("operations должен быть больше нуля")

        if not isinstance(self.fallback_used, bool):
            raise TypeError("fallback_used должен быть bool")

        if self.peak_memory_bytes is not None:
            if (
                isinstance(self.peak_memory_bytes, bool)
                or not isinstance(self.peak_memory_bytes, int)
            ):
                raise TypeError("peak_memory_bytes должен быть int или None")

            if self.peak_memory_bytes < 0:
                raise ValueError("peak_memory_bytes не может быть отрицательным")

        timings = (
            "compute_seconds",
            "transfer_seconds",
            "total_seconds",
        )

        for name in timings:
            value = getattr(self, name)

            if value is None:
                continue

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
            ):
                raise TypeError(f"{name} должен быть числом или None")

            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"{name} должен быть конечным неотрицательным числом"
                )

        if self.status is MeasurementStatus.SUCCESS:
            if self.actual_backend is not self.requested_backend:
                raise ValueError(
                    "SUCCESS требует совпадения requested_backend "
                    "и actual_backend"
                )

            if self.fallback_used:
                raise ValueError("SUCCESS не допускает fallback")

            if any(getattr(self, name) is None for name in timings):
                raise ValueError(
                    "SUCCESS требует все три фактических времени"
                )

            if self.compute_seconds <= 0:
                raise ValueError("compute_seconds должен быть больше нуля")

            if self.total_seconds + 1e-12 < (
                self.compute_seconds + self.transfer_seconds
            ):
                raise ValueError(
                    "total_seconds меньше суммы compute_seconds "
                    "и transfer_seconds"
                )

        if self.status is MeasurementStatus.FALLBACK and not self.fallback_used:
            raise ValueError("FALLBACK требует fallback_used=True")

        if self.status in (
            MeasurementStatus.SKIPPED,
            MeasurementStatus.FAILED,
        ) and self.actual_backend is not None:
            raise ValueError(
                "SKIPPED и FAILED не должны иметь actual_backend"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "requested_backend": self.requested_backend.value,
            "actual_backend": (
                self.actual_backend.value
                if self.actual_backend is not None
                else None
            ),
            "status": self.status.value,
            "repeat_index": self.repeat_index,
            "operations": self.operations,
            "input_bytes": self.input_bytes,
            "output_bytes": self.output_bytes,
            "compute_seconds": self.compute_seconds,
            "transfer_seconds": self.transfer_seconds,
            "total_seconds": self.total_seconds,
            "peak_memory_bytes": self.peak_memory_bytes,
            "fallback_used": self.fallback_used,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class PredictionComparison:
    """Сопоставление прогноза с фактической статистикой измерений."""

    case_id: str
    backend: ComputeBackend
    predicted_compute_seconds: float
    predicted_transfer_seconds: float
    predicted_total_seconds: float
    actual_compute_median: float
    actual_transfer_median: float
    actual_total_median: float
    actual_total_min: float
    actual_total_max: float
    actual_total_stddev: float
    sample_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.backend, ComputeBackend):
            raise TypeError("backend должен быть ComputeBackend")

        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError("case_id не может быть пустым")

        for name in (
            "predicted_compute_seconds",
            "predicted_transfer_seconds",
            "predicted_total_seconds",
            "actual_compute_median",
            "actual_transfer_median",
            "actual_total_median",
            "actual_total_min",
            "actual_total_max",
            "actual_total_stddev",
        ):
            value = getattr(self, name)

            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} должен быть числом")

            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"{name} должен быть конечным неотрицательным числом"
                )

        if self.predicted_total_seconds <= 0:
            raise ValueError("predicted_total_seconds должен быть больше нуля")

        if self.actual_total_median <= 0 or self.actual_total_min <= 0:
            raise ValueError("Фактическое время должно быть больше нуля")

        if self.actual_total_max < self.actual_total_min:
            raise ValueError("actual_total_max меньше actual_total_min")

        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
        ):
            raise TypeError("sample_count должен быть целым числом")

        if self.sample_count < 2:
            raise ValueError("sample_count должен быть не меньше 2")


@dataclass(frozen=True, slots=True)
class DecisionEvaluation:
    """Оценка решения планировщика по фактическим измерениям."""

    case_id: str
    selected_backend: ComputeBackend | None
    empirical_fastest_backend: ComputeBackend | None
    quality: DecisionQuality
    regret_seconds: float | None
    regret_ratio: float | None
    reason_code: str
    explanation: str
    comparable_backends: tuple[ComputeBackend, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.quality, DecisionQuality):
            raise TypeError("quality должен быть DecisionQuality")

        for name in ("selected_backend", "empirical_fastest_backend"):
            value = getattr(self, name)

            if value is not None and not isinstance(value, ComputeBackend):
                raise TypeError(f"{name} должен быть ComputeBackend или None")

        if not isinstance(self.comparable_backends, tuple):
            raise TypeError("comparable_backends должен быть tuple")

        if any(
            not isinstance(backend, ComputeBackend)
            for backend in self.comparable_backends
        ):
            raise TypeError(
                "comparable_backends должен содержать ComputeBackend"
            )

        if len(set(self.comparable_backends)) != len(
            self.comparable_backends
        ):
            raise ValueError("comparable_backends содержит дубли")

        for name in ("regret_seconds", "regret_ratio"):
            value = getattr(self, name)

            if value is None:
                continue

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
            ):
                raise TypeError(f"{name} должен быть числом или None")

            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"{name} должен быть конечным неотрицательным числом"
                )

        if not isinstance(self.reason_code, str):
            raise TypeError("reason_code должен быть строкой")

        if not isinstance(self.explanation, str):
            raise TypeError("explanation должен быть строкой")


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Полный результат эксперимента."""

    experiment_id: str
    created_at: str
    status: ExperimentStatus
    config: ExperimentConfig
    hardware: Mapping[str, Any]
    software: Mapping[str, Any]
    cases: tuple[ExperimentCase, ...]
    predictions: tuple[Mapping[str, Any], ...]
    measurements: tuple[ExperimentMeasurement, ...]
    comparisons: tuple[PredictionComparison, ...]
    decisions: tuple[DecisionEvaluation, ...]
    calibration_summary: tuple[Mapping[str, Any], ...]
    warnings: tuple[str, ...] = ()
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.experiment_id, str) or not self.experiment_id:
            raise ValueError("experiment_id не может быть пустым")

        if not isinstance(self.created_at, str) or not self.created_at:
            raise ValueError("created_at не может быть пустым")

        if not isinstance(self.status, ExperimentStatus):
            raise TypeError("status должен быть ExperimentStatus")

        if not isinstance(self.config, ExperimentConfig):
            raise TypeError("config должен быть ExperimentConfig")

        for name in (
            "cases",
            "predictions",
            "measurements",
            "comparisons",
            "decisions",
            "calibration_summary",
            "warnings",
        ):
            if not isinstance(getattr(self, name), tuple):
                raise TypeError(f"{name} должен быть tuple")

        if any(not isinstance(case, ExperimentCase) for case in self.cases):
            raise TypeError("cases должен содержать ExperimentCase")

        if any(
            not isinstance(item, ExperimentMeasurement)
            for item in self.measurements
        ):
            raise TypeError(
                "measurements должен содержать ExperimentMeasurement"
            )

        if any(
            not isinstance(item, PredictionComparison)
            for item in self.comparisons
        ):
            raise TypeError(
                "comparisons должен содержать PredictionComparison"
            )

        if any(
            not isinstance(item, DecisionEvaluation)
            for item in self.decisions
        ):
            raise TypeError(
                "decisions должен содержать DecisionEvaluation"
            )

    def as_dict(self) -> dict[str, Any]:
        """Возвращает JSON-совместимое представление результата."""
        return {
            "experiment_id": self.experiment_id,
            "created_at": self.created_at,
            "status": self.status.value,
            "experiment_type": "prediction_accuracy",
            "config": {
                "calibration_sizes": list(self.config.calibration_sizes),
                "validation_sizes": list(self.config.validation_sizes),
                "batch_sizes": list(self.config.batch_sizes),
                "repeats": self.config.repeats,
                "cuda_warmup": self.config.cuda_warmup,
                "random_seed": self.config.random_seed,
                "dtype": self.config.dtype,
                "minimum_gpu_speedup": self.config.minimum_gpu_speedup,
                "high_confidence_samples": (
                    self.config.high_confidence_samples
                ),
            },
            "hardware": dict(self.hardware),
            "software": dict(self.software),
            "cases": [case.as_dict() for case in self.cases],
            "predictions": [dict(item) for item in self.predictions],
            "measurements": [
                item.as_dict() for item in self.measurements
            ],
            "comparisons": [
                {
                    "case_id": item.case_id,
                    "backend": item.backend.value,
                    "predicted_compute_seconds": (
                        item.predicted_compute_seconds
                    ),
                    "predicted_transfer_seconds": (
                        item.predicted_transfer_seconds
                    ),
                    "predicted_total_seconds": (
                        item.predicted_total_seconds
                    ),
                    "actual_compute_median": item.actual_compute_median,
                    "actual_transfer_median": item.actual_transfer_median,
                    "actual_total_median": item.actual_total_median,
                    "actual_total_min": item.actual_total_min,
                    "actual_total_max": item.actual_total_max,
                    "actual_total_stddev": item.actual_total_stddev,
                    "sample_count": item.sample_count,
                }
                for item in self.comparisons
            ],
            "decisions": [
                {
                    "case_id": item.case_id,
                    "selected_backend": (
                        item.selected_backend.value
                        if item.selected_backend is not None
                        else None
                    ),
                    "empirical_fastest_backend": (
                        item.empirical_fastest_backend.value
                        if item.empirical_fastest_backend is not None
                        else None
                    ),
                    "quality": item.quality.value,
                    "regret_seconds": item.regret_seconds,
                    "regret_ratio": item.regret_ratio,
                    "reason_code": item.reason_code,
                    "explanation": item.explanation,
                    "comparable_backends": [
                        backend.value for backend in item.comparable_backends
                    ],
                }
                for item in self.decisions
            ],
            "calibration_summary": [
                dict(item) for item in self.calibration_summary
            ],
            "warnings": list(self.warnings),
            "error": self.error,
        }
