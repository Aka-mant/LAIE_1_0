
"""Модель прогнозирования производительности локальных вычислений.

Модель использует только фактические измерения. Она не переносит
калибровочные данные одного типа задач на другой тип задач.
"""

from dataclasses import dataclass
from math import isfinite
from statistics import median

from services.compute.scheduler.task_analyzer import TaskAnalysis
from services.compute.scheduler.task_types import ComputeBackend, TaskType


class InvalidPerformanceSampleError(ValueError):
    """Ошибка некорректного измерения производительности."""


class UnsupportedBackendError(ValueError):
    """Запрошенный backend не поддерживается задачей."""


@dataclass(frozen=True, slots=True)
class PerformanceSample:
    """Одно фактическое измерение выполнения задачи."""

    task_type: TaskType
    backend: ComputeBackend
    operations: int
    input_bytes: int
    output_bytes: int
    compute_seconds: float
    transfer_seconds: float
    total_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.task_type, TaskType):
            raise TypeError("task_type должен быть TaskType")

        if not isinstance(self.backend, ComputeBackend):
            raise TypeError("backend должен быть ComputeBackend")

        for name in ("operations", "input_bytes", "output_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} должен быть целым числом")
            if value < 0:
                raise ValueError(f"{name} не может быть отрицательным")

        if self.operations == 0:
            raise ValueError("operations должен быть больше нуля")

        for name in (
            "compute_seconds",
            "transfer_seconds",
            "total_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} должен быть числом")
            if not isfinite(value) or value < 0:
                raise ValueError(
                    f"{name} должен быть конечным неотрицательным числом"
                )

        if self.compute_seconds <= 0:
            raise ValueError("compute_seconds должен быть больше нуля")

        measured_minimum = (
            self.compute_seconds + self.transfer_seconds
        )
        if self.total_seconds + 1e-12 < measured_minimum:
            raise ValueError(
                "total_seconds не может быть меньше суммы "
                "compute_seconds и transfer_seconds"
            )

        if (
            self.backend is ComputeBackend.CPU
            and self.transfer_seconds != 0
        ):
            raise ValueError(
                "Для CPU transfer_seconds должен быть равен нулю"
            )


@dataclass(frozen=True, slots=True)
class PerformanceEstimate:
    """Прогноз времени выполнения задачи."""

    task_type: TaskType
    backend: ComputeBackend
    predicted_compute_seconds: float
    predicted_transfer_seconds: float
    predicted_overhead_seconds: float
    predicted_total_seconds: float
    sample_count: int
    confidence: str
    calibrated: bool = True

    def as_dict(self) -> dict:
        """Возвращает JSON-совместимое представление."""
        return {
            "task_type": self.task_type.value,
            "backend": self.backend.value,
            "predicted_compute_seconds": self.predicted_compute_seconds,
            "predicted_transfer_seconds": self.predicted_transfer_seconds,
            "predicted_overhead_seconds": self.predicted_overhead_seconds,
            "predicted_total_seconds": self.predicted_total_seconds,
            "sample_count": self.sample_count,
            "confidence": self.confidence,
            "calibrated": self.calibrated,
        }


class PerformanceModel:
    """Калибруемая модель производительности CPU и CUDA.

    Для каждой пары task_type/backend рассчитываются медианные
    удельные затраты времени:

    - секунды на одну операцию;
    - секунды на передачу одного байта;
    - постоянные накладные расходы.

    Модель не экстраполирует данные между разными типами задач.
    """

    def __init__(self, high_confidence_samples: int = 5) -> None:
        if (
            isinstance(high_confidence_samples, bool)
            or not isinstance(high_confidence_samples, int)
        ):
            raise TypeError(
                "high_confidence_samples должен быть целым числом"
            )
        if high_confidence_samples < 2:
            raise ValueError(
                "high_confidence_samples должен быть не меньше 2"
            )

        self.high_confidence_samples = high_confidence_samples
        self._samples: dict[
            tuple[TaskType, ComputeBackend],
            list[PerformanceSample],
        ] = {}

    @property
    def sample_count(self) -> int:
        """Общее число сохранённых измерений."""
        return sum(len(samples) for samples in self._samples.values())

    def add_sample(self, sample: PerformanceSample) -> None:
        """Добавляет фактическое измерение."""
        if not isinstance(sample, PerformanceSample):
            raise TypeError("sample должен быть PerformanceSample")

        key = (sample.task_type, sample.backend)
        self._samples.setdefault(key, []).append(sample)

    def get_samples(
        self,
        task_type: TaskType,
        backend: ComputeBackend,
    ) -> tuple[PerformanceSample, ...]:
        """Возвращает неизменяемый набор измерений."""
        self._validate_key(task_type, backend)
        return tuple(self._samples.get((task_type, backend), ()))

    def is_calibrated(
        self,
        task_type: TaskType,
        backend: ComputeBackend,
    ) -> bool:
        """Проверяет наличие калибровочных измерений."""
        return bool(self.get_samples(task_type, backend))


    def estimate(
        self,
        analysis: TaskAnalysis,
        backend: ComputeBackend,
        *,
        transfer_input_bytes: int | None = None,
        transfer_output_bytes: int | None = None,
    ) -> PerformanceEstimate | None:
        """Прогнозирует время выполнения по имеющимся измерениям.

        Объём передачи можно уточнить для конкретного сценария
        выполнения. Если он не задан, используются значения
        из TaskAnalysis.
        """
        if not isinstance(analysis, TaskAnalysis):
            raise TypeError("analysis должен быть TaskAnalysis")

        if not isinstance(backend, ComputeBackend):
            raise TypeError("backend должен быть ComputeBackend")

        if backend not in analysis.supported_backends:
            raise UnsupportedBackendError(
                f"{backend.value} не поддерживается задачей "
                f"{analysis.task_type.value}"
            )
        if (
            (transfer_input_bytes is None)
            != (transfer_output_bytes is None)
        ):
            raise ValueError(
                "Объёмы входной и выходной передачи "
                "должны задаваться вместе."
            )

        transfer_values = {
            "transfer_input_bytes": transfer_input_bytes,
            "transfer_output_bytes": transfer_output_bytes,
        }

        for name, value in transfer_values.items():
            if value is None:
                continue

            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} должен быть целым числом.")

            if value < 0:
                raise ValueError(f"{name} не может быть отрицательным.")

        if backend is ComputeBackend.CPU:
            if any(
                value is not None and value != 0
                for value in transfer_values.values()
            ):
                raise ValueError(
                    "Для CPU объём передачи данных должен быть равен нулю."
                )

        samples = self.get_samples(analysis.task_type, backend)

        if not samples:
            return None

        compute_rates = [
            sample.compute_seconds / sample.operations
            for sample in samples
        ]

        compute_seconds = (
            median(compute_rates) * analysis.estimated_operations
        )

        transfer_seconds = 0.0

        if backend is ComputeBackend.CUDA:
            transfer_rates = [
                sample.transfer_seconds
                / (sample.input_bytes + sample.output_bytes)
                for sample in samples
                if sample.input_bytes + sample.output_bytes > 0
                and sample.transfer_seconds > 0
            ]

            if transfer_rates:
                input_bytes = (
                    analysis.estimated_input_bytes
                    if transfer_input_bytes is None
                    else transfer_input_bytes
                )
                output_bytes = (
                    analysis.estimated_output_bytes
                    if transfer_output_bytes is None
                    else transfer_output_bytes
                )

                transfer_bytes = input_bytes + output_bytes

                transfer_seconds = (
                    median(transfer_rates) * transfer_bytes
                )

        overheads = [
            max(
                0.0,
                sample.total_seconds
                - sample.compute_seconds
                - sample.transfer_seconds,
            )
            for sample in samples
        ]

        overhead_seconds = median(overheads)

        total_seconds = (
            compute_seconds + transfer_seconds + overhead_seconds
        )

        count = len(samples)

        if count >= self.high_confidence_samples:
            confidence = "high"
        elif count >= 2:
            confidence = "medium"
        else:
            confidence = "low"

        return PerformanceEstimate(
            task_type=analysis.task_type,
            backend=backend,
            predicted_compute_seconds=compute_seconds,
            predicted_transfer_seconds=transfer_seconds,
            predicted_overhead_seconds=overhead_seconds,
            predicted_total_seconds=total_seconds,
            sample_count=count,
            confidence=confidence,
        )


    def estimate_supported(
        self,
        analysis: TaskAnalysis,
    ) -> tuple[PerformanceEstimate, ...]:
        """Возвращает прогнозы только для откалиброванных backend."""
        if not isinstance(analysis, TaskAnalysis):
            raise TypeError("analysis должен быть TaskAnalysis")

        estimates = []
        for backend in analysis.supported_backends:
            estimate = self.estimate(analysis, backend)
            if estimate is not None:
                estimates.append(estimate)

        return tuple(
            sorted(
                estimates,
                key=lambda item: item.predicted_total_seconds,
            )
        )

    def calibration_summary(self) -> list[dict]:
        """Возвращает сводку калибровки по типам задач и backend."""
        result = []

        for (task_type, backend), samples in sorted(
            self._samples.items(),
            key=lambda item: (
                item[0][0].value,
                item[0][1].value,
            ),
        ):
            result.append(
                {
                    "task_type": task_type.value,
                    "backend": backend.value,
                    "sample_count": len(samples),
                    "confidence": self._confidence(len(samples)),
                }
            )

        return result

    def _confidence(self, count: int) -> str:
        if count >= self.high_confidence_samples:
            return "high"
        if count >= 2:
            return "medium"
        return "low"

    @staticmethod
    def _validate_key(
        task_type: TaskType,
        backend: ComputeBackend,
    ) -> None:
        if not isinstance(task_type, TaskType):
            raise TypeError("task_type должен быть TaskType")
        if not isinstance(backend, ComputeBackend):
            raise TypeError("backend должен быть ComputeBackend")
