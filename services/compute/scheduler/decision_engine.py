
"""Движок принятия решений о выборе CPU или CUDA."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping

from .memory_estimator import MemoryEstimate
from .performance_model import PerformanceEstimate
from .task_analyzer import TaskAnalysis
from .task_types import ComputeBackend


class DecisionStatus(str, Enum):
    """Статус принятого вычислительного решения."""

    SELECTED = "selected"
    REJECTED = "rejected"


class DecisionReason(str, Enum):
    """Машиночитаемые причины выбора вычислительного устройства."""

    CPU_ONLY = "cpu_only"
    CPU_SAFE_FALLBACK = "cpu_safe_fallback"
    CPU_FASTER = "cpu_faster"
    GPU_FASTER = "gpu_faster"
    CPU_MEMORY_INSUFFICIENT = "cpu_memory_insufficient"
    GPU_MEMORY_INSUFFICIENT = "gpu_memory_insufficient"
    GPU_NOT_CALIBRATED = "gpu_not_calibrated"
    CPU_NOT_CALIBRATED = "cpu_not_calibrated"
    GPU_ADVANTAGE_TOO_SMALL = "gpu_advantage_too_small"
    NO_SUPPORTED_BACKEND = "no_supported_backend"
    NO_DEVICE_FITS = "no_device_fits"
    INVALID_ESTIMATE = "invalid_estimate"


@dataclass(frozen=True, slots=True)
class DecisionResult:
    """Результат выбора вычислительного устройства."""

    status: DecisionStatus
    selected_backend: ComputeBackend | None
    reason_code: DecisionReason
    explanation: str

    predicted_total_seconds: float | None
    confidence: str | None

    recommended_batch_size: int | None
    memory_required_bytes: int | None
    memory_usable_bytes: int | None

    evaluated_backends: tuple[ComputeBackend, ...]

    @property
    def accepted(self) -> bool:
        """Показывает, удалось ли выбрать устройство."""
        return self.status == DecisionStatus.SELECTED

    def as_dict(self) -> dict[str, Any]:
        """Возвращает JSON-совместимое представление решения."""
        result = asdict(self)
        result["status"] = self.status.value
        result["selected_backend"] = (
            self.selected_backend.value
            if self.selected_backend is not None
            else None
        )
        result["reason_code"] = self.reason_code.value
        result["evaluated_backends"] = [
            backend.value for backend in self.evaluated_backends
        ]
        return result


class DecisionEngine:
    """Выбирает CPU/CUDA с учётом памяти и прогноза производительности."""

    ACCEPTED_CONFIDENCE = frozenset({"medium", "high"})

    def __init__(
        self,
        *,
        minimum_gpu_speedup: float = 0.10,
    ) -> None:
        if isinstance(minimum_gpu_speedup, bool) or not isinstance(
            minimum_gpu_speedup,
            (int, float),
        ):
            raise TypeError("minimum_gpu_speedup должен быть числом.")

        if (
            not math.isfinite(minimum_gpu_speedup)
            or not 0 <= minimum_gpu_speedup < 1
        ):
            raise ValueError(
                "minimum_gpu_speedup должен находиться в диапазоне [0, 1)."
            )

        self.minimum_gpu_speedup = float(minimum_gpu_speedup)

    @staticmethod
    def _validate_inputs(
        analysis: TaskAnalysis,
        memory_estimates: Mapping[ComputeBackend, MemoryEstimate],
        performance_estimates: Mapping[
            ComputeBackend,
            PerformanceEstimate,
        ],
    ) -> None:
        if not isinstance(analysis, TaskAnalysis):
            raise TypeError("analysis должен быть TaskAnalysis.")

        if not isinstance(memory_estimates, Mapping):
            raise TypeError("memory_estimates должен быть Mapping.")

        if not isinstance(performance_estimates, Mapping):
            raise TypeError("performance_estimates должен быть Mapping.")

        for backend, estimate in memory_estimates.items():
            if not isinstance(backend, ComputeBackend):
                raise TypeError("Ключ memory_estimates должен быть ComputeBackend.")

            if not isinstance(estimate, MemoryEstimate):
                raise TypeError(
                    "Значения memory_estimates должны быть MemoryEstimate."
                )

            if estimate.backend != backend:
                raise ValueError("Backend MemoryEstimate не совпадает с ключом.")

            if estimate.task_type != analysis.task_type:
                raise ValueError("Тип задачи MemoryEstimate не совпадает.")

        for backend, estimate in performance_estimates.items():
            if not isinstance(backend, ComputeBackend):
                raise TypeError(
                    "Ключ performance_estimates должен быть ComputeBackend."
                )

            if not isinstance(estimate, PerformanceEstimate):
                raise TypeError(
                    "Значения performance_estimates должны быть "
                    "PerformanceEstimate."
                )

            if estimate.backend != backend:
                raise ValueError(
                    "Backend PerformanceEstimate не совпадает с ключом."
                )

            if estimate.task_type != analysis.task_type:
                raise ValueError("Тип задачи PerformanceEstimate не совпадает.")

    @staticmethod
    def _memory_is_usable(
        analysis: TaskAnalysis,
        estimate: MemoryEstimate | None,
    ) -> bool:
        if estimate is None:
            return False

        if estimate.fits:
            return True

        batch_size = estimate.recommended_batch_size
        batch_required = estimate.batch_required_bytes

        return (
            analysis.supports_batching
            and batch_size is not None
            and batch_size > 0
            and batch_required is not None
            and batch_required <= estimate.usable_bytes
        )

    def _valid_performance(
        self,
        estimate: PerformanceEstimate | None,
    ) -> bool:
        if estimate is None:
            return False

        if not estimate.calibrated:
            return False

        if estimate.confidence not in self.ACCEPTED_CONFIDENCE:
            return False

        return (
            math.isfinite(estimate.predicted_total_seconds)
            and estimate.predicted_total_seconds > 0
        )

    @staticmethod
    def _build_result(
        *,
        backend: ComputeBackend | None,
        reason: DecisionReason,
        explanation: str,
        memory: MemoryEstimate | None = None,
        performance: PerformanceEstimate | None = None,
        evaluated: tuple[ComputeBackend, ...] = (),
    ) -> DecisionResult:
        return DecisionResult(
            status=(
                DecisionStatus.SELECTED
                if backend is not None
                else DecisionStatus.REJECTED
            ),
            selected_backend=backend,
            reason_code=reason,
            explanation=explanation,
            predicted_total_seconds=(
                performance.predicted_total_seconds
                if performance is not None
                else None
            ),
            confidence=(
                performance.confidence
                if performance is not None
                else None
            ),
            recommended_batch_size=(
                memory.recommended_batch_size
                if memory is not None
                else None
            ),
            memory_required_bytes=(
                memory.required_bytes
                if memory is not None
                else None
            ),
            memory_usable_bytes=(
                memory.usable_bytes
                if memory is not None
                else None
            ),
            evaluated_backends=evaluated,
        )

    def decide(
        self,
        analysis: TaskAnalysis,
        memory_estimates: Mapping[ComputeBackend, MemoryEstimate],
        performance_estimates: Mapping[
            ComputeBackend,
            PerformanceEstimate,
        ],
    ) -> DecisionResult:
        """Выбирает backend на основании памяти и производительности."""
        self._validate_inputs(
            analysis,
            memory_estimates,
            performance_estimates,
        )

        supported = analysis.supported_backends
        evaluated = tuple(
            backend
            for backend in (ComputeBackend.CPU, ComputeBackend.CUDA)
            if backend in supported
        )

        if not evaluated:
            return self._build_result(
                backend=None,
                reason=DecisionReason.NO_SUPPORTED_BACKEND,
                explanation="Для задачи не определён поддерживаемый backend.",
            )

        cpu_supported = ComputeBackend.CPU in supported
        cuda_supported = ComputeBackend.CUDA in supported

        cpu_memory = memory_estimates.get(ComputeBackend.CPU)
        cuda_memory = memory_estimates.get(ComputeBackend.CUDA)

        cpu_fits = cpu_supported and self._memory_is_usable(
            analysis,
            cpu_memory,
        )
        cuda_fits = cuda_supported and self._memory_is_usable(
            analysis,
            cuda_memory,
        )

        cpu_performance = performance_estimates.get(ComputeBackend.CPU)
        cuda_performance = performance_estimates.get(ComputeBackend.CUDA)

        cpu_calibrated = self._valid_performance(cpu_performance)
        cuda_calibrated = self._valid_performance(cuda_performance)

        if cpu_supported and not cuda_supported:
            if not cpu_fits:
                return self._build_result(
                    backend=None,
                    reason=DecisionReason.NO_DEVICE_FITS,
                    explanation="CPU поддерживается, но памяти недостаточно.",
                    memory=cpu_memory,
                    evaluated=evaluated,
                )

            return self._build_result(
                backend=ComputeBackend.CPU,
                reason=DecisionReason.CPU_ONLY,
                explanation="Задача поддерживает только CPU; память достаточна.",
                memory=cpu_memory,
                performance=cpu_performance if cpu_calibrated else None,
                evaluated=evaluated,
            )

        if not cpu_supported and cuda_supported:
            if not cuda_fits:
                return self._build_result(
                    backend=None,
                    reason=DecisionReason.GPU_MEMORY_INSUFFICIENT,
                    explanation="CUDA поддерживается, но памяти недостаточно.",
                    memory=cuda_memory,
                    evaluated=evaluated,
                )

            if not cuda_calibrated:
                return self._build_result(
                    backend=None,
                    reason=DecisionReason.GPU_NOT_CALIBRATED,
                    explanation=(
                        "Для CUDA отсутствует надёжный калиброванный прогноз."
                    ),
                    memory=cuda_memory,
                    evaluated=evaluated,
                )

            return self._build_result(
                backend=ComputeBackend.CUDA,
                reason=DecisionReason.GPU_FASTER,
                explanation=(
                    "CUDA является единственным поддерживаемым backend; "
                    "памяти достаточно, прогноз откалиброван."
                ),
                memory=cuda_memory,
                performance=cuda_performance,
                evaluated=evaluated,
            )

        # Далее задача поддерживает одновременно CPU и CUDA.
        if cpu_fits and not cuda_fits:
            return self._build_result(
                backend=ComputeBackend.CPU,
                reason=DecisionReason.GPU_MEMORY_INSUFFICIENT,
                explanation=(
                    "CUDA не располагает достаточной памятью; "
                    "выбран безопасный CPU."
                ),
                memory=cpu_memory,
                performance=cpu_performance if cpu_calibrated else None,
                evaluated=evaluated,
            )

        if not cpu_fits and not cuda_fits:
            return self._build_result(
                backend=None,
                reason=DecisionReason.NO_DEVICE_FITS,
                explanation=(
                    "Ни CPU, ни CUDA не могут безопасно разместить задачу "
                    "или рекомендуемый пакет."
                ),
                evaluated=evaluated,
            )

        if not cpu_fits and cuda_fits:
            if not cuda_calibrated:
                return self._build_result(
                    backend=None,
                    reason=DecisionReason.GPU_NOT_CALIBRATED,
                    explanation=(
                        "CPU не располагает достаточной памятью, "
                        "а прогноз CUDA недостаточно надёжен."
                    ),
                    memory=cuda_memory,
                    evaluated=evaluated,
                )

            return self._build_result(
                backend=ComputeBackend.CUDA,
                reason=DecisionReason.GPU_FASTER,
                explanation=(
                    "CPU не располагает достаточной памятью; "
                    "CUDA подходит по памяти и имеет надёжный прогноз."
                ),
                memory=cuda_memory,
                performance=cuda_performance,
                evaluated=evaluated,
            )

        # Оба устройства подходят по памяти.
        if not cpu_calibrated:
            return self._build_result(
                backend=ComputeBackend.CPU,
                reason=DecisionReason.CPU_SAFE_FALLBACK,
                explanation=(
                    "CPU подходит по памяти. Нет надёжной оценки CPU, "
                    "поэтому преимущество CUDA нельзя доказать."
                ),
                memory=cpu_memory,
                evaluated=evaluated,
            )

        if not cuda_calibrated:
            return self._build_result(
                backend=ComputeBackend.CPU,
                reason=DecisionReason.GPU_NOT_CALIBRATED,
                explanation=(
                    "CUDA не имеет надёжного прогноза; выбран CPU."
                ),
                memory=cpu_memory,
                performance=cpu_performance,
                evaluated=evaluated,
            )

        cpu_time = cpu_performance.predicted_total_seconds
        cuda_time = cuda_performance.predicted_total_seconds

        required_gpu_time = cpu_time * (1 - self.minimum_gpu_speedup)

        if cuda_time <= required_gpu_time:
            return self._build_result(
                backend=ComputeBackend.CUDA,
                reason=DecisionReason.GPU_FASTER,
                explanation=(
                    f"Прогноз CUDA быстрее CPU минимум на "
                    f"{self.minimum_gpu_speedup:.0%}."
                ),
                memory=cuda_memory,
                performance=cuda_performance,
                evaluated=evaluated,
            )

        reason = (
            DecisionReason.CPU_FASTER
            if cpu_time <= cuda_time
            else DecisionReason.GPU_ADVANTAGE_TOO_SMALL
        )

        explanation = (
            "CPU быстрее по откалиброванному прогнозу."
            if reason == DecisionReason.CPU_FASTER
            else (
                "Преимущество CUDA недостаточно для установленного "
                "минимального порога; выбран CPU."
            )
        )

        return self._build_result(
            backend=ComputeBackend.CPU,
            reason=reason,
            explanation=explanation,
            memory=cpu_memory,
            performance=cpu_performance,
            evaluated=evaluated,
        )
