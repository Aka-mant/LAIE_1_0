"""Экспериментальные прогнозы CPU/CUDA для матричных вычислений LAIE.

Калькулятор воспроизводит расчёты AdaptiveMatrixService,
не вызывая его приватные методы и не изменяя производственный код.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from services.compute.scheduler.decision_engine import (
    DecisionEngine,
    DecisionResult,
)
from services.compute.scheduler.memory_estimator import (
    MemoryDeviceSnapshot,
    MemoryEstimate,
    MemoryEstimator,
)
from services.compute.scheduler.performance_model import (
    PerformanceEstimate,
    PerformanceModel,
)
from services.compute.scheduler.task_analyzer import (
    TaskAnalysis,
    TaskAnalyzer,
    TaskParameters,
)
from services.compute.scheduler.task_registry import (
    create_default_registry,
)
from services.compute.scheduler.task_types import (
    ComputeBackend,
    TaskType,
)


class PredictionExperimentCalculator:
    """Воспроизводит производственные прогнозы для эксперимента."""

    ELEMENT_SIZE = 8

    def __init__(
        self,
        *,
        analyzer: TaskAnalyzer,
        memory_estimator: MemoryEstimator,
        performance_model: PerformanceModel,
        decision_engine: DecisionEngine,
    ) -> None:
        dependencies = {
            "analyzer": (analyzer, TaskAnalyzer),
            "memory_estimator": (memory_estimator, MemoryEstimator),
            "performance_model": (performance_model, PerformanceModel),
            "decision_engine": (decision_engine, DecisionEngine),
        }

        for name, (value, expected_type) in dependencies.items():
            if not isinstance(value, expected_type):
                raise TypeError(
                    f"{name} должен быть {expected_type.__name__}."
                )

        self.analyzer = analyzer
        self.memory_estimator = memory_estimator
        self.performance_model = performance_model
        self.decision_engine = decision_engine

    @classmethod
    def create_default(
        cls,
        *,
        performance_model: PerformanceModel,
        minimum_gpu_speedup: float = 0.10,
        high_confidence_samples: int = 5,
    ) -> PredictionExperimentCalculator:
        """Создаёт калькулятор со штатным реестром матричных задач."""
        if not isinstance(performance_model, PerformanceModel):
            raise TypeError(
                "performance_model должен быть PerformanceModel."
            )

        registry = create_default_registry()

        analyzer = TaskAnalyzer(registry)
        memory_estimator = MemoryEstimator()
        decision_engine = DecisionEngine(
            minimum_gpu_speedup=minimum_gpu_speedup,
        )

        if (
            performance_model.high_confidence_samples
            != high_confidence_samples
        ):
            raise ValueError(
                "Порог высокой уверенности PerformanceModel "
                "не совпадает с настройкой эксперимента."
            )

        return cls(
            analyzer=analyzer,
            memory_estimator=memory_estimator,
            performance_model=performance_model,
            decision_engine=decision_engine,
        )

    @staticmethod
    def analyze(
        matrix_m: int,
        matrix_k: int,
        matrix_n: int,
    ) -> TaskAnalysis:
        """Анализирует матрицы через штатный TaskAnalyzer."""
        analyzer = TaskAnalyzer(create_default_registry())

        return analyzer.analyze(
            TaskType.MATRIX_MULTIPLICATION,
            TaskParameters(
                matrix_m=matrix_m,
                matrix_k=matrix_k,
                matrix_n=matrix_n,
            ),
        )

    def build_memory_analysis(
        self,
        analysis: TaskAnalysis,
        *,
        backend: ComputeBackend,
        rows: int,
        inner: int,
        columns: int,
        element_size: int = ELEMENT_SIZE,
    ) -> TaskAnalysis:
        """Строит производственную оценку памяти для backend."""
        if not isinstance(analysis, TaskAnalysis):
            raise TypeError("analysis должен быть TaskAnalysis.")

        if backend not in analysis.supported_backends:
            raise ValueError(
                f"Задача не поддерживает backend {backend.value}."
            )

        self._validate_dimensions(
            rows,
            inner,
            columns,
            element_size,
        )

        if backend is ComputeBackend.CUDA:
            input_bytes = (
                rows * inner + inner * columns
            ) * element_size

            output_bytes = rows * columns * element_size

        elif backend is ComputeBackend.CPU:
            input_bytes = 0
            output_bytes = 2 * rows * columns * element_size

        else:
            raise ValueError(f"Неподдерживаемый backend: {backend!r}")

        return replace(
            analysis,
            item_count=1,
            estimated_input_bytes=input_bytes,
            estimated_output_bytes=output_bytes,
            estimated_workspace_bytes=0,
            estimated_memory_bytes=input_bytes + output_bytes,
        )

    def estimate_cuda_batch(
        self,
        analysis: TaskAnalysis,
        snapshot: MemoryDeviceSnapshot,
        *,
        rows: int,
        inner: int,
        columns: int,
        element_size: int = ELEMENT_SIZE,
    ) -> tuple[int | None, int | None]:
        """Находит максимальный безопасный CUDA-пакет бинарным поиском."""
        if snapshot.backend is not ComputeBackend.CUDA:
            raise ValueError("Для поиска CUDA-пакета нужен CUDA snapshot.")

        self._validate_dimensions(
            rows,
            inner,
            columns,
            element_size,
        )

        low = 1
        high = rows
        best_rows: int | None = None
        best_required: int | None = None

        while low <= high:
            middle = (low + high) // 2

            input_bytes = (
                middle * inner + inner * columns
            ) * element_size

            output_bytes = middle * columns * element_size

            batch_analysis = replace(
                analysis,
                item_count=1,
                estimated_operations=(
                    2 * middle * inner * columns
                ),
                estimated_input_bytes=input_bytes,
                estimated_output_bytes=output_bytes,
                estimated_workspace_bytes=0,
                estimated_memory_bytes=input_bytes + output_bytes,
            )

            estimate = self.memory_estimator.estimate(
                batch_analysis,
                snapshot,
            )

            if estimate.fits:
                best_rows = middle
                best_required = estimate.required_bytes
                low = middle + 1
            else:
                high = middle - 1

        return best_rows, best_required

    def build_memory_estimates(
        self,
        analysis: TaskAnalysis,
        snapshots: tuple[MemoryDeviceSnapshot, ...],
        *,
        rows: int,
        inner: int,
        columns: int,
        element_size: int = ELEMENT_SIZE,
    ) -> dict[ComputeBackend, MemoryEstimate]:
        """Строит оценки памяти по CPU и CUDA."""
        self._validate_dimensions(
            rows,
            inner,
            columns,
            element_size,
        )

        estimates: dict[ComputeBackend, MemoryEstimate] = {}

        for snapshot in snapshots:
            backend = snapshot.backend

            if backend not in analysis.supported_backends:
                continue

            memory_analysis = self.build_memory_analysis(
                analysis,
                backend=backend,
                rows=rows,
                inner=inner,
                columns=columns,
                element_size=element_size,
            )

            estimate = self.memory_estimator.estimate(
                memory_analysis,
                snapshot,
            )

            if backend is ComputeBackend.CUDA:
                safe_rows, batch_required = self.estimate_cuda_batch(
                    analysis,
                    snapshot,
                    rows=rows,
                    inner=inner,
                    columns=columns,
                    element_size=element_size,
                )

                estimate = replace(
                    estimate,
                    recommended_batch_size=safe_rows,
                    batch_required_bytes=batch_required,
                )

            elif backend is ComputeBackend.CPU:
                estimate = replace(
                    estimate,
                    recommended_batch_size=None,
                    batch_required_bytes=None,
                )

            estimates[backend] = estimate

        return estimates

    def build_performance_estimates(
        self,
        analysis: TaskAnalysis,
        memory_estimates: Mapping[ComputeBackend, MemoryEstimate],
        *,
        rows: int,
        inner: int,
        columns: int,
        element_size: int = ELEMENT_SIZE,
        requested_batch_size: int,
    ) -> dict[ComputeBackend, PerformanceEstimate]:
        """Строит прогнозы с учётом реального объёма передачи данных."""
        self._validate_dimensions(
            rows,
            inner,
            columns,
            element_size,
        )

        if (
            isinstance(requested_batch_size, bool)
            or not isinstance(requested_batch_size, int)
        ):
            raise TypeError("requested_batch_size должен быть int.")

        if requested_batch_size <= 0:
            raise ValueError(
                "requested_batch_size должен быть положительным."
            )

        estimates: dict[ComputeBackend, PerformanceEstimate] = {}

        for backend in analysis.supported_backends:
            if backend is ComputeBackend.CUDA:
                memory_estimate = memory_estimates.get(backend)

                if memory_estimate is None:
                    continue

                safe_batch_size = (
                    memory_estimate.recommended_batch_size
                )

                if safe_batch_size is None or safe_batch_size <= 0:
                    continue

                actual_batch_size = min(
                    requested_batch_size,
                    safe_batch_size,
                    rows,
                )

                batch_count = (
                    rows + actual_batch_size - 1
                ) // actual_batch_size

                transfer_input_bytes = (
                    rows * inner
                    + batch_count * inner * columns
                ) * element_size

                transfer_output_bytes = (
                    rows * columns * element_size
                )

                estimate = self.performance_model.estimate(
                    analysis,
                    backend,
                    transfer_input_bytes=transfer_input_bytes,
                    transfer_output_bytes=transfer_output_bytes,
                )

            else:
                estimate = self.performance_model.estimate(
                    analysis,
                    backend,
                )

            if estimate is not None:
                estimates[backend] = estimate

        return estimates

    def decide(
        self,
        analysis: TaskAnalysis,
        memory_estimates: Mapping[ComputeBackend, MemoryEstimate],
        performance_estimates: Mapping[
            ComputeBackend,
            PerformanceEstimate,
        ],
    ) -> DecisionResult:
        """Получает решение штатного DecisionEngine."""
        return self.decision_engine.decide(
            analysis,
            dict(memory_estimates),
            dict(performance_estimates),
        )

    @staticmethod
    def clone_frozen_model(
        source: PerformanceModel,
    ) -> PerformanceModel:
        """Создаёт независимую модель из публичных калибровочных данных."""
        if not isinstance(source, PerformanceModel):
            raise TypeError("source должен быть PerformanceModel.")

        frozen = PerformanceModel(
            high_confidence_samples=source.high_confidence_samples,
        )

        for backend in (
            ComputeBackend.CPU,
            ComputeBackend.CUDA,
        ):
            samples = source.get_samples(
                TaskType.MATRIX_MULTIPLICATION,
                backend,
            )

            for sample in samples:
                frozen.add_sample(sample)

        return frozen

    @staticmethod
    def _validate_dimensions(
        rows: int,
        inner: int,
        columns: int,
        element_size: int,
    ) -> None:
        for name, value in (
            ("rows", rows),
            ("inner", inner),
            ("columns", columns),
            ("element_size", element_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} должен быть целым числом.")

            if value <= 0:
                raise ValueError(f"{name} должен быть положительным.")
