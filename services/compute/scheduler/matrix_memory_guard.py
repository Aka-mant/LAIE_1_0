
"""Контроль памяти для пакетного матричного умножения LAIE."""

from __future__ import annotations

from dataclasses import replace

from .hardware_monitor import HardwareMonitor
from .memory_estimator import MemoryEstimator
from .task_analyzer import TaskAnalysis
from .task_types import ComputeBackend, TaskType


class MatrixMemoryGuard:
    """Проверяет возможность выполнения каждого блока матрицы."""

    def __init__(
        self,
        analysis: TaskAnalysis,
        *,
        rows: int,
        inner: int,
        columns: int,
        element_size: int,
        hardware_monitor: HardwareMonitor | None = None,
        memory_estimator: MemoryEstimator | None = None,
    ) -> None:
        if not isinstance(analysis, TaskAnalysis):
            raise TypeError("analysis должен быть TaskAnalysis.")

        if analysis.task_type is not TaskType.MATRIX_MULTIPLICATION:
            raise ValueError(
                "MatrixMemoryGuard предназначен только "
                "для матричного умножения."
            )

        dimensions = {
            "rows": rows,
            "inner": inner,
            "columns": columns,
            "element_size": element_size,
        }

        for name, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} должен быть целым числом.")

            if value <= 0:
                raise ValueError(f"{name} должен быть больше нуля.")

        if not analysis.supports_batching:
            raise ValueError(
                "Анализ задачи должен поддерживать пакетирование."
            )

        if hardware_monitor is not None and not isinstance(
            hardware_monitor,
            HardwareMonitor,
        ):
            raise TypeError(
                "hardware_monitor должен быть HardwareMonitor."
            )

        if memory_estimator is not None and not isinstance(
            memory_estimator,
            MemoryEstimator,
        ):
            raise TypeError(
                "memory_estimator должен быть MemoryEstimator."
            )

        self.analysis = analysis
        self.rows = rows
        self.inner = inner
        self.columns = columns
        self.element_size = element_size

        self.hardware_monitor = (
            hardware_monitor or HardwareMonitor()
        )
        self.memory_estimator = (
            memory_estimator or MemoryEstimator()
        )

    def __call__(
        self,
        start: int,
        stop: int,
        backend: ComputeBackend,
    ) -> bool:
        """Проверяет память перед выполнением пакета.

        Возвращает False при недостатке памяти или невозможности
        получить достоверный снимок оборудования.
        """

        if not isinstance(backend, ComputeBackend):
            raise TypeError("backend должен быть ComputeBackend.")

        if isinstance(start, bool) or not isinstance(start, int):
            raise TypeError("start должен быть целым числом.")

        if isinstance(stop, bool) or not isinstance(stop, int):
            raise TypeError("stop должен быть целым числом.")

        if not 0 <= start < stop <= self.rows:
            raise ValueError("Некорректный диапазон строк пакета.")

        if backend not in self.analysis.supported_backends:
            return False

        try:
            hardware = self.hardware_monitor.inspect()
        except (OSError, RuntimeError):
            return False

        snapshots = hardware.memory_snapshots()

        snapshot = next(
            (
                item
                for item in snapshots
                if item.backend is backend
            ),
            None,
        )

        if snapshot is None:
            return False

        batch_rows = stop - start

        memory_analysis = self._build_batch_analysis(
            start=start,
            batch_rows=batch_rows,
            backend=backend,
        )

        estimate = self.memory_estimator.estimate(
            memory_analysis,
            snapshot,
        )

        return estimate.fits

    def _build_batch_analysis(
        self,
        *,
        start: int,
        batch_rows: int,
        backend: ComputeBackend,
    ) -> TaskAnalysis:
        """Создаёт специализированную оценку памяти для пакета."""

        if backend is ComputeBackend.CUDA:
            # На GPU находятся блок left, полная right и результат блока.
            input_bytes = (
                batch_rows * self.inner
                + self.inner * self.columns
            ) * self.element_size

            output_bytes = (
                batch_rows * self.columns * self.element_size
            )

        else:
            # Входные матрицы уже размещены в RAM.
            # Учитываем будущие результаты пакетов и полную
            # итоговую матрицу, создаваемую при torch.cat().
            remaining_rows = self.rows - start

            input_bytes = 0

            output_bytes = (
                remaining_rows * self.columns
                + self.rows * self.columns
            ) * self.element_size

        estimated_memory = input_bytes + output_bytes

        return replace(
            self.analysis,
            item_count=1,
            estimated_operations=(
                2 * batch_rows * self.inner * self.columns
            ),
            estimated_input_bytes=input_bytes,
            estimated_output_bytes=output_bytes,
            estimated_workspace_bytes=0,
            estimated_memory_bytes=estimated_memory,
        )
