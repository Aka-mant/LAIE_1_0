
"""Тесты сквозного координатора матричного умножения LAIE."""

from __future__ import annotations

from threading import Event

import pytest

from services.compute.scheduler.execution_engine import ExecutionEngine
from services.compute.scheduler.pytorch_adapter import (
    CUDAExecutionError,

    PyTorchCUDAAdapter,
)
from services.compute.scheduler.task_analyzer import TaskAnalyzer
from services.compute.scheduler.task_registry import create_default_registry
from services.compute.scheduler.task_types import ComputeBackend

from services.compute.scheduler.matrix_coordinator import (
    MatrixMultiplicationCoordinator,
)

torch = pytest.importorskip("torch")


@pytest.fixture
def analyzer() -> TaskAnalyzer:
    """Создаёт анализатор со стандартным реестром."""
    return TaskAnalyzer(create_default_registry())


@pytest.fixture
def adapter() -> PyTorchCUDAAdapter:
    """Создаёт настоящий PyTorch-адаптер."""
    return PyTorchCUDAAdapter()


@pytest.fixture
def coordinator(
    analyzer: TaskAnalyzer,
    adapter: PyTorchCUDAAdapter,
) -> MatrixMultiplicationCoordinator:
    """Создаёт координатор матричных операций."""
    return MatrixMultiplicationCoordinator(
        analyzer=analyzer,
        adapter=adapter,
    )


@pytest.fixture
def matrices():
    """Создаёт тестовые матрицы с известными размерами."""
    left = torch.arange(
        35,
        dtype=torch.float64,
    ).reshape(7, 5)

    right = torch.arange(
        20,
        dtype=torch.float64,
    ).reshape(5, 4)

    return left, right


def test_cpu_matrix_multiplication_is_successful(
    coordinator,
    matrices,
):
    """Проверяет успешное умножение матриц на CPU."""
    left, right = matrices

    report = coordinator.execute(
        left,
        right,
        requested_backend=ComputeBackend.CPU,
        batch_size=3,
    )

    expected = torch.matmul(left, right)

    assert report.successful
    assert report.actual_backend is ComputeBackend.CPU
    assert report.execution.result_validated
    assert torch.allclose(report.execution.result, expected)


def test_matrix_is_split_into_correct_row_batches(
    coordinator,
    matrices,
):
    """Проверяет пакетирование, включая последний неполный пакет."""
    left, right = matrices

    report = coordinator.execute(
        left,
        right,
        requested_backend=ComputeBackend.CPU,
        batch_size=3,
    )

    assert report.successful
    assert report.execution.metrics.batch_count == 3
    assert report.execution.metrics.processed_items == 7

    assert len(report.batch_executions) == 3

    assert [
        item.result.shape[0]
        for item in report.batch_executions
    ] == [3, 3, 1]


def test_analysis_uses_matrix_dimensions(
    coordinator,
    matrices,
):
    """Проверяет передачу размеров M, K и N в анализатор."""
    left, right = matrices

    report = coordinator.execute(
        left,
        right,
        requested_backend=ComputeBackend.CPU,
        batch_size=2,
    )

    assert report.analysis.item_count == 7 * 5 * 4
    assert report.analysis.estimated_operations == 2 * 7 * 5 * 4


def test_execution_request_uses_row_count(
    coordinator,
    matrices,
):
    """Проверяет, что исполнитель обрабатывает строки, а не FLOPs."""
    left, right = matrices

    report = coordinator.execute(
        left,
        right,
        requested_backend=ComputeBackend.CPU,
        batch_size=2,
    )

    assert report.execution.metrics.processed_items == left.shape[0]
    assert report.execution.metrics.batch_count == 4


def test_report_contains_aggregated_measurements(
    coordinator,
    matrices,
):
    """Проверяет суммирование фактических метрик пакетов."""
    left, right = matrices

    report = coordinator.execute(
        left,
        right,
        requested_backend=ComputeBackend.CPU,
        batch_size=2,
    )

    assert report.successful
    assert report.compute_seconds > 0
    assert report.transfer_seconds == 0
    assert report.adapter_total_seconds > 0
    assert report.operations == 2 * 7 * 5 * 4


def test_report_peak_memory_is_none_for_cpu(
    coordinator,
    matrices,
):
    """Проверяет отсутствие CUDA-памяти в CPU-режиме."""
    left, right = matrices

    report = coordinator.execute(
        left,
        right,
        requested_backend=ComputeBackend.CPU,
        batch_size=3,
    )

    assert report.successful
    assert report.peak_memory_bytes is None


def test_float32_matrices_are_supported(
    coordinator,
):
    """Проверяет работу с float32."""
    left = torch.randn(5, 4, dtype=torch.float32)
    right = torch.randn(4, 3, dtype=torch.float32)

    report = coordinator.execute(
        left,
        right,
        requested_backend=ComputeBackend.CPU,
        batch_size=2,
    )

    expected = torch.matmul(left, right)

    assert report.successful
    assert report.execution.result.dtype == torch.float32
    assert torch.allclose(
        report.execution.result,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "batch_size",
    [0, -1],
)
def test_invalid_batch_size_is_rejected(
    coordinator,
    matrices,
    batch_size,
):
    """Отклоняет нулевой и отрицательный размер пакета."""
    left, right = matrices

    with pytest.raises(ValueError):
        coordinator.execute(
            left,
            right,
            requested_backend=ComputeBackend.CPU,
            batch_size=batch_size,
        )


def test_boolean_batch_size_is_rejected(
    coordinator,
    matrices,
):
    """Отклоняет bool вместо целого размера пакета."""
    left, right = matrices

    with pytest.raises(TypeError):
        coordinator.execute(
            left,
            right,
            requested_backend=ComputeBackend.CPU,
            batch_size=True,
        )


def test_float_batch_size_is_rejected(
    coordinator,
    matrices,
):
    """Отклоняет дробный размер пакета."""
    left, right = matrices

    with pytest.raises(TypeError):
        coordinator.execute(
            left,
            right,
            requested_backend=ComputeBackend.CPU,
            batch_size=2.5,
        )


def test_invalid_backend_is_rejected(
    coordinator,
    matrices,
):
    """Отклоняет строку вместо ComputeBackend."""
    left, right = matrices

    with pytest.raises(TypeError):
        coordinator.execute(
            left,
            right,
            requested_backend="cpu",
            batch_size=2,
        )


def test_one_dimensional_matrix_is_rejected(
    coordinator,
):
    """Отклоняет одномерные входные тензоры."""
    left = torch.randn(5)
    right = torch.randn(5, 3)

    with pytest.raises(ValueError):
        coordinator.execute(
            left,
            right,
            requested_backend=ComputeBackend.CPU,
            batch_size=2,
        )


def test_incompatible_matrix_dimensions_are_rejected(
    coordinator,
):
    """Отклоняет матрицы с несовместимыми размерами."""
    left = torch.randn(5, 4)
    right = torch.randn(3, 2)

    with pytest.raises(ValueError):
        coordinator.execute(
            left,
            right,
            requested_backend=ComputeBackend.CPU,
            batch_size=2,
        )


def test_different_dtypes_are_rejected(
    coordinator,
):
    """Отклоняет матрицы с разными типами данных."""
    left = torch.randn(5, 4, dtype=torch.float32)
    right = torch.randn(4, 3, dtype=torch.float64)

    with pytest.raises(ValueError):
        coordinator.execute(
            left,
            right,
            requested_backend=ComputeBackend.CPU,
            batch_size=2,
        )


def test_empty_dimension_is_rejected(
    coordinator,
):
    """Отклоняет матрицы с пустыми измерениями."""
    left = torch.empty((0, 4))
    right = torch.empty((4, 3))

    with pytest.raises(ValueError):
        coordinator.execute(
            left,
            right,
            requested_backend=ComputeBackend.CPU,
            batch_size=2,
        )


def test_cuda_execution_when_available(
    coordinator,
    matrices,
):
    """Проверяет реальное умножение на CUDA при её доступности."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA недоступна.")

    left, right = matrices

    report = coordinator.execute(
        left,
        right,
        requested_backend=ComputeBackend.CUDA,
        batch_size=3,
    )

    expected = torch.matmul(left, right)

    assert report.successful
    assert report.actual_backend is ComputeBackend.CUDA
    assert report.execution.result.device.type == "cpu"
    assert torch.allclose(
        report.execution.result,
        expected,
        rtol=1e-8,
        atol=1e-8,
    )
    assert report.transfer_seconds > 0
    assert report.peak_memory_bytes is not None


def test_cuda_failure_uses_cpu_fallback(
    analyzer,
    matrices,
):
    """Проверяет переход на CPU после ошибки CUDA."""

    class FailingCUDAAdapter(PyTorchCUDAAdapter):
        """Имитирует отказ CUDA, сохраняя реальный CPU-адаптер."""

        def matmul(
            self,
            left,
            right,
            *,
            backend,
        ):
            if backend is ComputeBackend.CUDA:
                raise CUDAExecutionError(
                    "Тестовая ошибка CUDA."
                )

            return super().matmul(
                left,
                right,
                backend=backend,
            )

    adapter = FailingCUDAAdapter()

    coordinator = MatrixMultiplicationCoordinator(
        analyzer=analyzer,
        adapter=adapter,
        execution_engine=ExecutionEngine(
            cpu_fallback_enabled=True,
        ),
    )

    left, right = matrices

    report = coordinator.execute(
        left,
        right,
        requested_backend=ComputeBackend.CUDA,
        batch_size=3,
    )

    expected = torch.matmul(left, right)

    assert report.successful
    assert report.execution.fallback_used
    assert report.actual_backend is ComputeBackend.CPU
    assert torch.allclose(report.execution.result, expected)
    assert len(report.batch_executions) == 3


def test_cpu_fallback_can_be_disabled(
    analyzer,
    matrices,
):
    """Проверяет, что отключённый fallback не запускает CPU."""

    class FailingCUDAAdapter(PyTorchCUDAAdapter):
        """Всегда имитирует отказ CUDA."""

        def matmul(
            self,
            left,
            right,
            *,
            backend,
        ):
            if backend is ComputeBackend.CUDA:
                raise CUDAExecutionError(
                    "Тестовая ошибка CUDA."
                )

            return super().matmul(
                left,
                right,
                backend=backend,
            )

    coordinator = MatrixMultiplicationCoordinator(
        analyzer=analyzer,
        adapter=FailingCUDAAdapter(),
        execution_engine=ExecutionEngine(
            cpu_fallback_enabled=False,
        ),
    )

    left, right = matrices

    report = coordinator.execute(
        left,
        right,
        requested_backend=ComputeBackend.CUDA,
        batch_size=3,
    )

    assert not report.successful
    assert not report.execution.fallback_used
    assert report.execution.result is None


def test_cancellation_before_execution(
    coordinator,
    matrices,
):
    """Проверяет отмену до выполнения первого пакета."""
    left, right = matrices
    cancellation_event = Event()
    cancellation_event.set()

    report = coordinator.execute(
        left,
        right,
        requested_backend=ComputeBackend.CPU,
        batch_size=2,
        cancellation_event=cancellation_event,
    )

    assert not report.successful
    assert report.execution.status.value == "cancelled"
    assert report.execution.metrics.batch_count == 0
    assert report.batch_executions == ()


def test_constructor_rejects_invalid_analyzer(
    adapter,
):
    """Проверяет тип анализатора."""
    with pytest.raises(TypeError):
        MatrixMultiplicationCoordinator(
            analyzer=object(),
            adapter=adapter,
        )


def test_constructor_rejects_invalid_adapter(
    analyzer,
):
    """Проверяет тип адаптера."""
    with pytest.raises(TypeError):
        MatrixMultiplicationCoordinator(
            analyzer=analyzer,
            adapter=object(),
        )

def test_coordinator_checks_memory_before_every_batch(
    coordinator,
    matrices,
    monkeypatch,
):
    """Проверяет передачу MatrixMemoryGuard координатором исполнителю."""
    checks = []

    class RecordingMemoryGuard:
        """Записывает вызовы защиты памяти."""

        def __init__(
            self,
            analysis,
            *,
            rows,
            inner,
            columns,
            element_size,
        ):
            self.analysis = analysis
            self.rows = rows
            self.inner = inner
            self.columns = columns
            self.element_size = element_size

        def __call__(self, start, stop, backend):
            checks.append((start, stop, backend))
            return True

    monkeypatch.setattr(
        "services.compute.scheduler.matrix_coordinator.MatrixMemoryGuard",
        RecordingMemoryGuard,
    )

    left, right = matrices

    report = coordinator.execute(
        left,
        right,
        requested_backend=ComputeBackend.CPU,
        batch_size=3,
    )

    assert report.successful
    assert checks == [
        (0, 3, ComputeBackend.CPU),
        (3, 6, ComputeBackend.CPU),
        (6, 7, ComputeBackend.CPU),
    ]


def test_cuda_mid_execution_failure_restarts_from_first_batch(
    analyzer,
    matrices,
):
    """Проверяет полный перезапуск матричной операции на CPU."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA недоступна.")

    class FailingSecondCUDABatchAdapter(PyTorchCUDAAdapter):
        """Имитирует сбой CUDA на второй партии."""

        def __init__(self):
            super().__init__()
            self.calls = []
            self.cuda_batches = 0

        def matmul(
            self,
            left,
            right,
            *,
            backend,
        ):
            first_value = float(left[0, 0].item())

            self.calls.append((backend, first_value))

            if backend is ComputeBackend.CUDA:
                self.cuda_batches += 1

                if self.cuda_batches == 2:
                    raise CUDAExecutionError(
                        "Тестовый сбой CUDA на второй партии."
                    )

            return super().matmul(
                left,
                right,
                backend=backend,
            )

    adapter = FailingSecondCUDABatchAdapter()

    coordinator = MatrixMultiplicationCoordinator(
        analyzer=analyzer,
        adapter=adapter,
        execution_engine=ExecutionEngine(
            cpu_fallback_enabled=True,
        ),
    )

    left, right = matrices

    report = coordinator.execute(
        left,
        right,
        requested_backend=ComputeBackend.CUDA,
        batch_size=3,
    )

    expected = torch.matmul(left, right)

    assert report.successful
    assert report.execution.fallback_used
    assert report.actual_backend is ComputeBackend.CPU

    assert adapter.calls == [
        (ComputeBackend.CUDA, 0.0),
        (ComputeBackend.CUDA, 15.0),
        (ComputeBackend.CPU, 0.0),
        (ComputeBackend.CPU, 15.0),
        (ComputeBackend.CPU, 30.0),
    ]

    assert torch.allclose(
        report.execution.result,
        expected,
    )


def test_cpu_fallback_report_contains_only_cpu_measurements(
    analyzer,
    matrices,
):
    """Исключает частичные CUDA-измерения из успешного CPU-отчёта."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA недоступна.")

    class FailingSecondCUDABatchAdapter(PyTorchCUDAAdapter):
        """Завершает первую CUDA-партию и имитирует сбой второй."""

        def __init__(self):
            super().__init__()
            self.cuda_batches = 0

        def matmul(
            self,
            left,
            right,
            *,
            backend,
        ):
            if backend is ComputeBackend.CUDA:
                self.cuda_batches += 1

                if self.cuda_batches == 2:
                    raise CUDAExecutionError(
                        "Тестовый сбой CUDA."
                    )

            return super().matmul(
                left,
                right,
                backend=backend,
            )

    coordinator = MatrixMultiplicationCoordinator(
        analyzer=analyzer,
        adapter=FailingSecondCUDABatchAdapter(),
        execution_engine=ExecutionEngine(
            cpu_fallback_enabled=True,
        ),
    )

    left, right = matrices

    report = coordinator.execute(
        left,
        right,
        requested_backend=ComputeBackend.CUDA,
        batch_size=3,
    )

    assert report.successful
    assert report.actual_backend is ComputeBackend.CPU
    assert report.execution.fallback_used

    assert len(report.batch_executions) == 3

    assert all(
        measurement.backend is ComputeBackend.CPU
        for measurement in report.batch_executions
    )

    assert report.compute_seconds == pytest.approx(
        sum(
            measurement.compute_seconds
            for measurement in report.batch_executions
        )
    )

    assert report.transfer_seconds == 0

    assert report.adapter_total_seconds == pytest.approx(
        sum(
            measurement.total_seconds
            for measurement in report.batch_executions
        )
    )
