
"""Адаптивный выбор CPU/CUDA для матричного умножения LAIE."""

from __future__ import annotations

from dataclasses import dataclass, replace

from typing import Any

from .decision_engine import DecisionEngine, DecisionResult
from .hardware_monitor import HardwareMonitor, HardwareSnapshot
from .matrix_coordinator import (
    MatrixExecutionReport,
    MatrixMultiplicationCoordinator,
)
from .matrix_performance_trainer import MatrixPerformanceTrainer
from .memory_estimator import (
    MemoryEstimate,
    MemoryEstimator,
    MemoryDeviceSnapshot,
)
from .performance_model import PerformanceEstimate, PerformanceModel
from .task_analyzer import TaskAnalysis, TaskAnalyzer, TaskParameters
from .task_types import ComputeBackend, TaskType


@dataclass(frozen=True, slots=True)
class AdaptiveMatrixReport:
    """Полный отчёт об адаптивном выполнении матричной задачи."""

    analysis: TaskAnalysis
    hardware: HardwareSnapshot
    memory_estimates: dict[ComputeBackend, MemoryEstimate]
    performance_estimates: dict[ComputeBackend, PerformanceEstimate]
    decision: DecisionResult
    execution: MatrixExecutionReport | None
    training_recorded: bool

    @property
    def successful(self) -> bool:
        """Показывает, завершилось ли вычисление успешно."""
        return (
            self.execution is not None
            and self.execution.successful
        )

    @property
    def actual_backend(self) -> ComputeBackend | None:
        """Возвращает фактически использованный backend."""
        if self.execution is None:
            return None

        return self.execution.actual_backend

    def as_dict(self) -> dict[str, Any]:
        """Возвращает JSON-совместимое представление отчёта."""
        return {
            "task_type": self.analysis.task_type.value,
            "hardware": self.hardware.as_dict(),
            "decision": self.decision.as_dict(),
            "memory_estimates": {
                backend.value: estimate.as_dict()
                for backend, estimate in self.memory_estimates.items()
            },
            "performance_estimates": {
                backend.value: estimate.as_dict()
                for backend, estimate in self.performance_estimates.items()
            },
            "execution": (
                {
                    "successful": self.execution.successful,
                    "actual_backend": (
                        self.execution.actual_backend.value
                        if self.execution.actual_backend is not None
                        else None
                    ),
                    "compute_seconds": self.execution.compute_seconds,
                    "transfer_seconds": self.execution.transfer_seconds,
                    "adapter_total_seconds": (
                        self.execution.adapter_total_seconds
                    ),
                    "peak_memory_bytes": self.execution.peak_memory_bytes,
                    "operations": self.execution.operations,
                }
                if self.execution is not None
                else None
            ),
            "training_recorded": self.training_recorded,
        }


class AdaptiveMatrixService:
    """Координирует анализ, выбор оборудования, выполнение и обучение."""

    def __init__(
        self,
        *,
        analyzer: TaskAnalyzer,
        hardware_monitor: HardwareMonitor,
        memory_estimator: MemoryEstimator,
        performance_model: PerformanceModel,
        decision_engine: DecisionEngine,
        coordinator: MatrixMultiplicationCoordinator,
        trainer: MatrixPerformanceTrainer,
    ) -> None:
        dependencies = {
            "analyzer": (analyzer, TaskAnalyzer),
            "hardware_monitor": (hardware_monitor, HardwareMonitor),
            "memory_estimator": (memory_estimator, MemoryEstimator),
            "performance_model": (performance_model, PerformanceModel),
            "decision_engine": (decision_engine, DecisionEngine),
            "coordinator": (
                coordinator,
                MatrixMultiplicationCoordinator,
            ),
            "trainer": (trainer, MatrixPerformanceTrainer),
        }

        for name, (value, expected_type) in dependencies.items():
            if not isinstance(value, expected_type):
                raise TypeError(
                    f"{name} должен быть {expected_type.__name__}."
                )

        self.analyzer = analyzer
        self.hardware_monitor = hardware_monitor
        self.memory_estimator = memory_estimator
        self.performance_model = performance_model
        self.decision_engine = decision_engine
        self.coordinator = coordinator
        self.trainer = trainer

    @staticmethod
    def _validate_batch_size(batch_size: int) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size должен быть целым числом.")

        if batch_size <= 0:
            raise ValueError("batch_size должен быть больше нуля.")

    @staticmethod
    def _validate_dimensions(
        left: Any,
        right: Any,
    ) -> tuple[int, int, int, int]:
        """Проверяет форму и тип данных входных матриц."""

        try:
            left_shape = tuple(left.shape)
            right_shape = tuple(right.shape)
        except (AttributeError, TypeError) as exc:
            raise TypeError(
                "left и right должны быть двумерными тензорами."
            ) from exc

        if len(left_shape) != 2 or len(right_shape) != 2:
            raise ValueError("Обе матрицы должны быть двумерными.")

        rows, inner = left_shape
        right_inner, columns = right_shape

        dimensions = (rows, inner, right_inner, columns)

        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in dimensions
        ):
            raise ValueError(
                "Размерности матриц должны быть положительными целыми."
            )

        if inner != right_inner:
            raise ValueError(
                "Количество столбцов left должно совпадать "
                "с количеством строк right."
            )

        if left.dtype != right.dtype:
            raise ValueError("Типы данных матриц должны совпадать.")

        try:
            element_size = int(left.element_size())
        except (AttributeError, TypeError, ValueError) as exc:
            raise TypeError(
                "Не удалось определить размер элемента матрицы."
            ) from exc

        if element_size <= 0:
            raise ValueError("Размер элемента должен быть больше нуля.")

        return rows, inner, columns, element_size

    def _build_memory_analysis(
        self,
        analysis: TaskAnalysis,
        *,
        backend: ComputeBackend,
        rows: int,
        inner: int,
        columns: int,
        element_size: int,
    ) -> TaskAnalysis:
        """Создаёт матричную оценку памяти для указанного backend."""

        if backend is ComputeBackend.CUDA:
            input_bytes = (
                rows * inner + inner * columns
            ) * element_size

            output_bytes = (
                rows * columns * element_size
            )
        else:
            # Входные тензоры уже находятся в RAM.
            # Учитываем накопленные блоки и итоговый torch.cat().
            input_bytes = 0
            output_bytes = (
                2 * rows * columns * element_size
            )

        return replace(
            analysis,
            item_count=1,
            estimated_input_bytes=input_bytes,
            estimated_output_bytes=output_bytes,
            estimated_workspace_bytes=0,
            estimated_memory_bytes=input_bytes + output_bytes,
        )

    def _estimate_cuda_batch(
        self,
        analysis: TaskAnalysis,
        snapshot: MemoryDeviceSnapshot,
        *,
        rows: int,
        inner: int,
        columns: int,
        element_size: int,
    ) -> tuple[int | None, int | None]:
        """Находит максимальный безопасный размер CUDA-пакета."""

        low = 1
        high = rows
        best_rows: int | None = None
        best_required: int | None = None

        while low <= high:
            middle = (low + high) // 2

            batch_analysis = replace(
                analysis,
                item_count=1,
                estimated_operations=(
                    2 * middle * inner * columns
                ),
                estimated_input_bytes=(
                    middle * inner + inner * columns
                ) * element_size,
                estimated_output_bytes=(
                    middle * columns * element_size
                ),
                estimated_workspace_bytes=0,
            )

            batch_analysis = replace(
                batch_analysis,
                estimated_memory_bytes=(
                    batch_analysis.estimated_input_bytes
                    + batch_analysis.estimated_output_bytes
                ),
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


    def _build_memory_estimates(
        self,
        analysis: TaskAnalysis,
        snapshots: tuple[MemoryDeviceSnapshot, ...],
        *,
        rows: int,
        inner: int,
        columns: int,
        element_size: int,
    ) -> dict[ComputeBackend, MemoryEstimate]:
        """Строит оценки памяти с учётом матричной геометрии."""

        estimates: dict[ComputeBackend, MemoryEstimate] = {}

        for snapshot in snapshots:
            backend = snapshot.backend

            if backend not in analysis.supported_backends:
                continue

            memory_analysis = self._build_memory_analysis(
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
                safe_rows, batch_required = self._estimate_cuda_batch(
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



    def _build_performance_estimates(
        self,
        analysis: TaskAnalysis,
        memory_estimates: dict[ComputeBackend, Any],
        *,
        rows: int,
        inner: int,
        columns: int,
        element_size: int,
        requested_batch_size: int,
    ) -> dict[ComputeBackend, Any]:
        """Строит прогнозы с учётом фактического объёма передачи данных."""
        estimates = {}

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


    def execute(
        self,
        left: Any,
        right: Any,
        *,
        batch_size: int = 256,
        cancellation_event: Any | None = None,
    ) -> AdaptiveMatrixReport:
        """Адаптивно выбирает CPU/CUDA и выполняет матричное умножение."""

        self._validate_batch_size(batch_size)

        if cancellation_event is not None and not (
                callable(getattr(cancellation_event, "is_set", None))
                and callable(getattr(cancellation_event, "set", None))
        ):
            raise TypeError(
                "cancellation_event должен поддерживать интерфейс threading.Event."
            )

        rows, inner, columns, element_size = self._validate_dimensions(
            left,
            right,
        )

        analysis = self.analyzer.analyze(
            TaskType.MATRIX_MULTIPLICATION,
            TaskParameters(
                matrix_m=rows,
                matrix_k=inner,
                matrix_n=columns,
            ),
        )

        if analysis.task_type is not TaskType.MATRIX_MULTIPLICATION:
            raise ValueError(
                "TaskAnalyzer вернул анализ не матричного умножения."
            )

        hardware = self.hardware_monitor.inspect()
        snapshots = hardware.memory_snapshots()

        memory_estimates = self._build_memory_estimates(
            analysis,
            snapshots,
            rows=rows,
            inner=inner,
            columns=columns,
            element_size=element_size,
        )

        performance_estimates = self._build_performance_estimates(
            analysis,
            memory_estimates,
            rows=rows,
            inner=inner,
            columns=columns,
            element_size=
            element_size,
            requested_batch_size=batch_size,
        )

        decision = self.decision_engine.decide(
            analysis,
            memory_estimates,
            performance_estimates,
        )

        if not decision.accepted:
            return AdaptiveMatrixReport(
                analysis=analysis,
                hardware=hardware,
                memory_estimates=memory_estimates,
                performance_estimates=performance_estimates,
                decision=decision,
                execution=None,
                training_recorded=False,
            )

        selected_backend = decision.selected_backend

        if selected_backend is None:
            raise RuntimeError(
                "DecisionEngine вернул accepted без выбранного backend."
            )

        selected_memory = memory_estimates[selected_backend]

        actual_batch_size = batch_size

        if selected_backend is ComputeBackend.CUDA:
            safe_batch_size = selected_memory.recommended_batch_size

            if safe_batch_size is None or safe_batch_size <= 0:
                raise RuntimeError(
                    "DecisionEngine выбрал CUDA без безопасного размера пакета."
                )

            actual_batch_size = min(
                batch_size,
                safe_batch_size,
                rows,
            )
        else:
            actual_batch_size = min(batch_size, rows)

        execution = self.coordinator.execute(
            left,
            right,
            requested_backend=selected_backend,
            batch_size=actual_batch_size,
            cancellation_event=cancellation_event,
        )

        training_recorded = False

        if execution.successful:
            training_recorded = self.trainer.record(execution)

        return AdaptiveMatrixReport(
            analysis=analysis,
            hardware=hardware,
            memory_estimates=memory_estimates,
            performance_estimates=performance_estimates,
            decision=decision,
            execution=execution,
            training_recorded=training_recorded,
        )
