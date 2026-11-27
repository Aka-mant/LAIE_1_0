
"""Координатор сквозного выполнения матричного умножения LAIE."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from threading import Event
from typing import Any

from .execution_engine import ExecutionEngine, ExecutionRequest
from .execution_types import ExecutionResult, ExecutionStatus
from .pytorch_adapter import MatmulExecution, PyTorchCUDAAdapter
from .task_analyzer import TaskAnalysis, TaskAnalyzer, TaskParameters
from .task_types import ComputeBackend, TaskType
from .matrix_memory_guard import MatrixMemoryGuard


@dataclass(frozen=True, slots=True)
class MatrixExecutionReport:
    """Отчёт о выполнении матричного умножения."""

    analysis: TaskAnalysis
    execution: ExecutionResult
    batch_executions: tuple[MatmulExecution, ...]

    @property
    def successful(self) -> bool:
        """Проверяет успешность выполнения."""
        return self.execution.status is ExecutionStatus.SUCCESS

    @property
    def actual_backend(self) -> ComputeBackend | None:
        """Возвращает фактически использованное устройство."""
        return self.execution.actual_backend

    @property
    def compute_seconds(self) -> float:
        """Возвращает суммарное время вычислений адаптера."""
        return sum(
            item.compute_seconds for item in self.batch_executions
        )

    @property
    def transfer_seconds(self) -> float:
        """Возвращает суммарное время передачи данных."""
        return sum(
            item.transfer_seconds for item in self.batch_executions
        )

    @property
    def adapter_total_seconds(self) -> float:
        """Возвращает сумму времени операций адаптера."""
        return sum(
            item.total_seconds for item in self.batch_executions
        )

    @property
    def peak_memory_bytes(self) -> int | None:
        """Возвращает максимальное измеренное потребление памяти."""
        values = [
            item.peak_memory_bytes
            for item in self.batch_executions
            if item.peak_memory_bytes is not None
        ]

        return max(values) if values else None

    @property
    def operations(self) -> int:
        """Возвращает суммарное число измеренных операций."""
        return sum(item.operations for item in self.batch_executions)


class MatrixMultiplicationCoordinator:
    """Связывает анализ, пакетное выполнение и PyTorch-адаптер."""

    def __init__(
        self,
        analyzer: TaskAnalyzer,
        adapter: PyTorchCUDAAdapter,
        execution_engine: ExecutionEngine | None = None,
        torch_module: Any | None = None,
    ) -> None:
        if not isinstance(analyzer, TaskAnalyzer):
            raise TypeError("analyzer должен быть TaskAnalyzer.")

        if not isinstance(adapter, PyTorchCUDAAdapter):
            raise TypeError("adapter должен быть PyTorchCUDAAdapter.")

        if execution_engine is not None and not isinstance(
            execution_engine,
            ExecutionEngine,
        ):
            raise TypeError(
                "execution_engine должен быть ExecutionEngine."
            )

        self.analyzer = analyzer
        self.adapter = adapter
        self.execution_engine = (
            execution_engine or ExecutionEngine()
        )
        self._torch_module = torch_module

    def execute(
        self,
        left: Any,
        right: Any,
        *,
        requested_backend: ComputeBackend,
        batch_size: int,
        cancellation_event: Event | None = None,
    ) -> MatrixExecutionReport:
        """Выполняет матричное умножение пакетами строк.

        Матрицы должны быть двумерными CPU-тензорами.
        Пакетирование выполняется по строкам первой матрицы.
        """

        if not isinstance(requested_backend, ComputeBackend):
            raise TypeError(
                "requested_backend должен быть ComputeBackend."
            )

        if isinstance(batch_size, bool) or not isinstance(
            batch_size,
            int,
        ):
            raise TypeError("batch_size должен быть целым числом.")

        if batch_size <= 0:
            raise ValueError("batch_size должен быть больше нуля.")

        self._validate_matrix_shapes(left, right)

        rows = int(left.shape[0])
        inner = int(left.shape[1])
        columns = int(right.shape[1])

        analysis = self.analyzer.analyze(
            TaskType.MATRIX_MULTIPLICATION,
            TaskParameters(
                matrix_m=rows,
                matrix_k=inner,
                matrix_n=columns,
            ),
        )

        request = ExecutionRequest(
            task_type=TaskType.MATRIX_MULTIPLICATION,
            item_count=rows,
            batch_size=batch_size,
            requested_backend=requested_backend,
            retry_safe=True,
        )

        measurements: dict[ComputeBackend, list[MatmulExecution]] = {
            ComputeBackend.CPU: [],
            ComputeBackend.CUDA: [],
        }

        def batch_executor(
            start: int,
            stop: int,
            backend: ComputeBackend,
        ) -> MatmulExecution:
            """Выполняет один блок строк на выбранном устройстве."""
            batch = left[start:stop]

            measurement = self.adapter.matmul(
                batch,
                right,
                backend=backend,
            )

            measurements[backend].append(measurement)
            return measurement

        def result_combiner(
            batch_results: list[MatmulExecution],
        ) -> Any:
            """Объединяет результаты пакетов по строкам."""
            if not batch_results:
                raise ValueError(
                    "Невозможно объединить пустой набор пакетов."
                )

            torch = self._load_torch()

            return torch.cat(
                [item.result for item in batch_results],
                dim=0,
            )

        def result_validator(
            result: Any,
            processed_items: int,
        ) -> bool:
            """Проверяет форму и тип итоговой матрицы."""
            if not hasattr(result, "shape"):
                return False

            if len(result.shape) != 2:
                return False

            if tuple(result.shape) != (rows, columns):
                return False

            if result.dtype != left.dtype:
                return False

            if result.device.type != "cpu":
                return False

            return processed_items == rows

        memory_guard = MatrixMemoryGuard(
            analysis,
            rows=rows,
            inner=inner,
            columns=columns,
            element_size=left.element_size(),
        )

        execution = self.execution_engine.execute(
            request,
            batch_executor,
            result_combiner=result_combiner,
            result_validator=result_validator,
            cancellation_event=cancellation_event,
            memory_guard=memory_guard,
        )

        successful_measurements = (
            tuple(measurements[execution.actual_backend])
            if execution.actual_backend is not None
            else ()
        )

        return MatrixExecutionReport(
            analysis=analysis,
            execution=execution,
            batch_executions=successful_measurements,
        )

    @staticmethod
    def _validate_matrix_shapes(
        left: Any,
        right: Any,
    ) -> None:
        """Проверяет базовую структуру входных матриц."""

        if not hasattr(left, "shape") or not hasattr(right, "shape"):
            raise TypeError(
                "left и right должны быть тензорами PyTorch."
            )

        if len(left.shape) != 2 or len(right.shape) != 2:
            raise ValueError(
                "Обе входные матрицы должны быть двумерными."
            )

        if left.device.type != "cpu" or right.device.type != "cpu":
            raise ValueError(
                "Входные матрицы должны находиться на CPU."
            )

        if left.shape[0] <= 0 or left.shape[1] <= 0:
            raise ValueError(
                "Размеры первой матрицы должны быть положительными."
            )

        if right.shape[0] <= 0 or right.shape[1] <= 0:
            raise ValueError(
                "Размеры второй матрицы должны быть положительными."
            )

        if left.shape[1] != right.shape[0]:
            raise ValueError(
                "Количество столбцов left должно совпадать "
                "с количеством строк right."
            )

        if left.dtype != right.dtype:
            raise ValueError(
                "Входные матрицы должны иметь одинаковый dtype."
            )

    def _load_torch(self) -> Any:
        """Лениво загружает PyTorch для объединения результатов."""
        if self._torch_module is None:
            try:
                self._torch_module = importlib.import_module("torch")
            except ImportError as exc:
                raise RuntimeError(
                    "Для объединения матриц необходим PyTorch."
                ) from exc

        return self._torch_module
