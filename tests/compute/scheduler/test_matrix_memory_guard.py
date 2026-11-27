
"""Тесты контроля памяти матричного умножения LAIE."""

from __future__ import annotations

import pytest

from services.compute.scheduler.hardware_monitor import (
    HardwareMonitor,
    HardwareSnapshot,
)
from services.compute.scheduler.matrix_memory_guard import (
    MatrixMemoryGuard,
)
from services.compute.scheduler.memory_estimator import (
    MemoryEstimator,
)
from services.compute.scheduler.task_analyzer import (
    TaskAnalyzer,
    TaskParameters,
)
from services.compute.scheduler.task_registry import create_default_registry
from services.compute.scheduler.task_types import (
    ComputeBackend,
    TaskType,
)


@pytest.fixture
def analyzer() -> TaskAnalyzer:
    """Создаёт анализатор стандартных задач."""
    return TaskAnalyzer(create_default_registry())


@pytest.fixture
def analysis(analyzer: TaskAnalyzer):
    """Создаёт анализ матричного умножения 7x5 на 5x4."""
    return analyzer.analyze(
        TaskType.MATRIX_MULTIPLICATION,
        TaskParameters(
            matrix_m=7,
            matrix_k=5,
            matrix_n=4,
        ),
    )


@pytest.fixture
def estimator() -> MemoryEstimator:
    """Создаёт оценщик с нулевыми резервами для точных тестов."""
    return MemoryEstimator(
        safety_margin_ratio=0,
        cpu_reserve_bytes=0,
        cuda_reserve_bytes=0,
        cpu_runtime_overhead_bytes=0,
        cuda_runtime_overhead_bytes=0,
    )


class StaticHardwareMonitor(HardwareMonitor):
    """Возвращает заданный снимок оборудования."""

    def __init__(self, snapshot: HardwareSnapshot):
        super().__init__()
        self.snapshot = snapshot
        self.calls = 0

    def inspect(self) -> HardwareSnapshot:
        self.calls += 1
        return self.snapshot


def make_snapshot(
    *,
    ram_free: int = 10_000,
    cuda_available: bool = True,
    cuda_free: int = 10_000,
) -> HardwareSnapshot:
    """Создаёт тестовый снимок оборудования."""
    return HardwareSnapshot(
        cpu_name="Test CPU",
        cpu_logical_cores=8,
        ram_total_bytes=20_000,
        ram_available_bytes=ram_free,
        cuda_available=cuda_available,
        cuda_device_name="Test CUDA" if cuda_available else None,
        cuda_total_bytes=20_000 if cuda_available else None,
        cuda_free_bytes=cuda_free if cuda_available else None,
        cuda_reason=None if cuda_available else "CUDA unavailable",
    )


@pytest.fixture
def guard(analysis, estimator):
    """Создаёт контроль памяти для тестовой матрицы."""
    monitor = StaticHardwareMonitor(make_snapshot())

    return MatrixMemoryGuard(
        analysis,
        rows=7,
        inner=5,
        columns=4,
        element_size=8,
        hardware_monitor=monitor,
        memory_estimator=estimator,
    )


def test_cpu_guard_accepts_sufficient_memory(
    analysis,
    estimator,
):
    """Разрешает пакет при достаточной RAM."""
    monitor = StaticHardwareMonitor(
        make_snapshot(ram_free=448)
    )

    guard = MatrixMemoryGuard(
        analysis,
        rows=7,
        inner=5,
        columns=4,
        element_size=8,
        hardware_monitor=monitor,
        memory_estimator=estimator,
    )

    assert guard(0, 3, ComputeBackend.CPU)


def test_cpu_guard_rejects_insufficient_memory(
    analysis,
    estimator,
):
    """Запрещает пакет при недостаточной RAM."""
    monitor = StaticHardwareMonitor(
        make_snapshot(ram_free=447)
    )

    guard = MatrixMemoryGuard(
        analysis,
        rows=7,
        inner=5,
        columns=4,
        element_size=8,
        hardware_monitor=monitor,
        memory_estimator=estimator,
    )

    assert not guard(0, 3, ComputeBackend.CPU)


def test_cuda_guard_calculates_batch_memory(
    analysis,
    estimator,
):
    """Учитывает блок left, полную right и результат блока."""
    monitor = StaticHardwareMonitor(
        make_snapshot(cuda_free=376)
    )

    guard = MatrixMemoryGuard(
        analysis,
        rows=7,
        inner=5,
        columns=4,
        element_size=8,
        hardware_monitor=monitor,
        memory_estimator=estimator,
    )

    assert guard(0, 3, ComputeBackend.CUDA)


def test_cuda_guard_rejects_insufficient_vram(
    analysis,
    estimator,
):
    """Запрещает CUDA-пакет при недостатке VRAM."""
    monitor = StaticHardwareMonitor(
        make_snapshot(cuda_free=375)
    )

    guard = MatrixMemoryGuard(
        analysis,
        rows=7,
        inner=5,
        columns=4,
        element_size=8,
        hardware_monitor=monitor,
        memory_estimator=estimator,
    )

    assert not guard(0, 3, ComputeBackend.CUDA)


def test_cuda_guard_rejects_unavailable_device(
    analysis,
    estimator,
):
    """Запрещает CUDA, если устройство недоступно."""
    monitor = StaticHardwareMonitor(
        make_snapshot(cuda_available=False)
    )

    guard = MatrixMemoryGuard(
        analysis,
        rows=7,
        inner=5,
        columns=4,
        element_size=8,
        hardware_monitor=monitor,
        memory_estimator=estimator,
    )

    assert not guard(0, 3, ComputeBackend.CUDA)


def test_guard_refreshes_hardware_snapshot(
    guard,
):
    """Получает новый снимок оборудования при каждом вызове."""
    assert guard(0, 2, ComputeBackend.CPU)
    assert guard(2, 4, ComputeBackend.CPU)

    assert guard.hardware_monitor.calls == 2


def test_cpu_memory_estimate_changes_with_remaining_rows(
    analysis,
    estimator,
):
    """Учитывает уменьшение объёма ещё не созданных результатов."""
    enough_for_first = StaticHardwareMonitor(
        make_snapshot(ram_free=448)
    )

    enough_for_last = StaticHardwareMonitor(
        make_snapshot(ram_free=256)
    )

    first_guard = MatrixMemoryGuard(
        analysis,
        rows=7,
        inner=5,
        columns=4,
        element_size=8,
        hardware_monitor=enough_for_first,
        memory_estimator=estimator,
    )

    last_guard = MatrixMemoryGuard(
        analysis,
        rows=7,
        inner=5,
        columns=4,
        element_size=8,
        hardware_monitor=enough_for_last,
        memory_estimator=estimator,
    )

    assert first_guard(0, 3, ComputeBackend.CPU)
    assert last_guard(6, 7, ComputeBackend.CPU)


def test_guard_rejects_invalid_range(
    guard,
):
    """Отклоняет диапазон строк вне матрицы."""
    with pytest.raises(ValueError):
        guard(-1, 2, ComputeBackend.CPU)

    with pytest.raises(ValueError):
        guard(2, 2, ComputeBackend.CPU)

    with pytest.raises(ValueError):
        guard(0, 8, ComputeBackend.CPU)


@pytest.mark.parametrize(
    ("start", "stop"),
    [
        (True, 2),
        (0, False),
        (1.5, 3),
        (0, 2.5),
    ],
)
def test_guard_rejects_invalid_range_types(
    guard,
    start,
    stop,
):
    """Отклоняет некорректные типы границ диапазона."""
    with pytest.raises(TypeError):
        guard(start, stop, ComputeBackend.CPU)


def test_guard_rejects_invalid_backend(
    guard,
):
    """Отклоняет неизвестный backend."""
    with pytest.raises(TypeError):
        guard(0, 2, "cpu")


def test_guard_fails_closed_when_monitor_fails(
    analysis,
    estimator,
):
    """Запрещает пакет, если мониторинг памяти завершился ошибкой."""

    class FailedMonitor(HardwareMonitor):
        def inspect(self):
            raise OSError("Тестовая ошибка мониторинга.")

    guard = MatrixMemoryGuard(
        analysis,
        rows=7,
        inner=5,
        columns=4,
        element_size=8,
        hardware_monitor=FailedMonitor(),
        memory_estimator=estimator,
    )

    assert not guard(0, 2, ComputeBackend.CPU)


def test_constructor_rejects_invalid_analysis(
    estimator,
):
    """Проверяет тип входного анализа."""
    with pytest.raises(TypeError):
        MatrixMemoryGuard(
            object(),
            rows=7,
            inner=5,
            columns=4,
            element_size=8,
            memory_estimator=estimator,
        )


@pytest.mark.parametrize(
    "field",
    ["rows", "inner", "columns", "element_size"],
)
def test_constructor_rejects_non_positive_dimensions(
    analysis,
    estimator,
    field,
):
    """Отклоняет нулевые размеры матрицы и элемента."""
    dimensions = {
        "rows": 7,
        "inner": 5,
        "columns": 4,
        "element_size": 8,
    }
    dimensions[field] = 0

    with pytest.raises(ValueError):
        MatrixMemoryGuard(
            analysis,
            hardware_monitor=StaticHardwareMonitor(make_snapshot()),
            memory_estimator=estimator,
            **dimensions,
        )


def test_constructor_rejects_boolean_dimension(
    analysis,
    estimator,
):
    """Отклоняет bool вместо размера."""
    with pytest.raises(TypeError):
        MatrixMemoryGuard(
            analysis,
            rows=True,
            inner=5,
            columns=4,
            element_size=8,
            memory_estimator=estimator,
        )


def test_constructor_rejects_wrong_monitor(
    analysis,
    estimator,
):
    """Проверяет тип монитора оборудования."""
    with pytest.raises(TypeError):
        MatrixMemoryGuard(
            analysis,
            rows=7,
            inner=5,
            columns=4,
            element_size=8,
            hardware_monitor=object(),
            memory_estimator=estimator,
        )


def test_constructor_rejects_wrong_estimator(
    analysis,
):
    """Проверяет тип оценщика памяти."""
    with pytest.raises(TypeError):
        MatrixMemoryGuard(
            analysis,
            rows=7,
            inner=5,
            columns=4,
            element_size=8,
            memory_estimator=object(),
        )
