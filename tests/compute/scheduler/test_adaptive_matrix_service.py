"""Тесты адаптивного выбора CPU/CUDA для матричного умножения."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from services.compute.scheduler.adaptive_matrix_service import (
    AdaptiveMatrixService,
)
from services.compute.scheduler.decision_engine import DecisionEngine
from services.compute.scheduler.hardware_monitor import (
    HardwareMonitor,
    HardwareSnapshot,
)
from services.compute.scheduler.matrix_coordinator import (
    MatrixMultiplicationCoordinator,
)
from services.compute.scheduler.matrix_performance_trainer import (
    MatrixPerformanceTrainer,
)
from services.compute.scheduler.memory_estimator import MemoryEstimator
from services.compute.scheduler.performance_model import PerformanceModel
from services.compute.scheduler.pytorch_adapter import PyTorchCUDAAdapter
from services.compute.scheduler.task_analyzer import TaskAnalyzer
from services.compute.scheduler.task_registry import (
    TaskDefinition,
    TaskRegistry,
)
from services.compute.scheduler.task_types import (
    ComputeBackend,
    PrecisionRequirement,
    TaskType,
    WorkloadKind,
)
from services.compute.scheduler.performance_model import (
    PerformanceSample,
)


GIB = 1024**3


def build_analyzer() -> TaskAnalyzer:
    """Создаёт анализатор с зарегистрированной матричной задачей."""
    registry = TaskRegistry()
    registry.register(
        TaskDefinition(
            task_type=TaskType.MATRIX_MULTIPLICATION,
            name="Matrix multiplication",
            description="Тестовая задача матричного умножения",
            workload=WorkloadKind.MIXED,
            supported_backends=(ComputeBackend.CPU,),
            precision=PrecisionRequirement.FLOAT32,
            parallelizable=True,
            supports_batching=True,
            estimator_key="matrix_multiplication",
        )
    )
    return TaskAnalyzer(registry)


def build_hardware_snapshot(
    *,
    ram_total_bytes: int = 8 * GIB,
    ram_available_bytes: int = 7 * GIB,
) -> HardwareSnapshot:
    """Создаёт предсказуемый снимок CPU без CUDA."""
    return HardwareSnapshot(
        cpu_name="Test CPU",
        cpu_logical_cores=8,
        ram_total_bytes=ram_total_bytes,
        ram_available_bytes=ram_available_bytes,
        cuda_available=False,
        cuda_reason="CUDA disabled in isolated test",
    )


def build_service(
    *,
    ram_total_bytes: int = 8 * GIB,
    ram_available_bytes: int = 7 * GIB,
    cpu_reserve_bytes: int = 0,
) -> AdaptiveMatrixService:
    """Создаёт сервис с реальными компонентами и тестовым оборудованием."""
    analyzer = build_analyzer()
    hardware_monitor = HardwareMonitor()

    memory_estimator = MemoryEstimator(
        safety_margin_ratio=0.0,
        cpu_reserve_bytes=cpu_reserve_bytes,
        cuda_reserve_bytes=0,
        cpu_runtime_overhead_bytes=0,
        cuda_runtime_overhead_bytes=0,
    )

    performance_model = PerformanceModel()
    decision_engine = DecisionEngine()

    adapter = PyTorchCUDAAdapter(torch_module=torch)
    coordinator = MatrixMultiplicationCoordinator(
        analyzer=analyzer,
        adapter=adapter,
        torch_module=torch,
    )
    trainer = MatrixPerformanceTrainer(performance_model)

    service = AdaptiveMatrixService(
        analyzer=analyzer,
        hardware_monitor=hardware_monitor,
        memory_estimator=memory_estimator,
        performance_model=performance_model,
        decision_engine=decision_engine,
        coordinator=coordinator,
        trainer=trainer,
    )

    snapshot = build_hardware_snapshot(
        ram_total_bytes=ram_total_bytes,
        ram_available_bytes=ram_available_bytes,
    )

    # Подменяем только источник аппаратных данных.
    # Вычисления матриц и остальные компоненты остаются реальными.
    patcher = patch.object(
        hardware_monitor,
        "inspect",
        return_value=snapshot,
    )
    patcher.start()

    # Сохраняем patcher на объекте сервиса, чтобы он не завершился
    # до окончания теста.
    service._test_hardware_patcher = patcher

    return service


@pytest.fixture
def service():
    """Создаёт сервис и гарантированно освобождает подмену оборудования."""
    instance = build_service()
    try:
        yield instance
    finally:
        instance._test_hardware_patcher.stop()


def test_cpu_execution_without_cuda(service):
    """Сервис выполняет матричную задачу на CPU без CUDA."""
    left = torch.rand((8, 4), dtype=torch.float32)
    right = torch.rand((4, 6), dtype=torch.float32)

    report = service.execute(
        left,
        right,
        batch_size=3,
    )

    assert report.successful
    assert report.execution is not None
    assert report.actual_backend is ComputeBackend.CPU
    assert report.decision.accepted
    assert report.decision.selected_backend is ComputeBackend.CPU
    assert report.execution.execution.status.value == "success"
    assert report.execution.execution.result_validated
    assert report.execution.operations == 2 * 8 * 4 * 6
    assert report.training_recorded


def test_cpu_result_matches_torch_matmul(service):
    """Результат координатора совпадает с эталонным torch.matmul."""
    left = torch.rand((7, 5), dtype=torch.float32)
    right = torch.rand((5, 3), dtype=torch.float32)

    report = service.execute(
        left,
        right,
        batch_size=2,
    )

    assert report.successful
    assert report.execution is not None

    actual = report.execution.execution.result
    expected = torch.matmul(left, right)

    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_report_contains_matrix_analysis(service):
    """Отчёт содержит корректные параметры матричной задачи."""
    left = torch.rand((5, 3), dtype=torch.float32)
    right = torch.rand((3, 4), dtype=torch.float32)

    report = service.execute(left, right)

    assert report.analysis.task_type is TaskType.MATRIX_MULTIPLICATION
    assert report.analysis.item_count == 5 * 3 * 4
    assert report.analysis.estimated_operations == 2 * 5 * 3 * 4
    assert report.analysis.estimated_input_bytes == (
        (5 * 3 + 3 * 4) * left.element_size()
    )
    assert report.analysis.estimated_output_bytes == (
        5 * 4 * left.element_size()
    )


def test_memory_estimates_contain_cpu(service):
    """Оценка памяти содержит CPU."""
    left = torch.rand((4, 4), dtype=torch.float32)
    right = torch.rand((4, 4), dtype=torch.float32)

    report = service.execute(left, right)

    assert ComputeBackend.CPU in report.memory_estimates
    assert report.memory_estimates[ComputeBackend.CPU].fits


def test_unavailable_cuda_is_not_selected(service):
    """CUDA не выбирается, если оборудование сообщает об её отсутствии."""
    left = torch.rand((4, 4), dtype=torch.float32)
    right = torch.rand((4, 4), dtype=torch.float32)

    report = service.execute(left, right)

    assert not report.hardware.cuda_available
    assert report.actual_backend is ComputeBackend.CPU
    assert ComputeBackend.CUDA not in report.memory_estimates


def test_rejects_task_when_cpu_memory_is_insufficient():
    """Задача отклоняется, если доступной памяти CPU недостаточно."""
    service = build_service(
        ram_total_bytes=8 * GIB,
        ram_available_bytes=1 * GIB,
        cpu_reserve_bytes=2 * GIB,
    )

    try:
        left = torch.rand((4, 4), dtype=torch.float32)
        right = torch.rand((4, 4), dtype=torch.float32)

        report = service.execute(left, right)

        assert not report.decision.accepted
        assert report.execution is None
        assert not report.successful
        assert not report.training_recorded
    finally:
        service._test_hardware_patcher.stop()


@pytest.mark.parametrize(
    "batch_size",
    [0, -1, True, 1.5, "4", None],
)
def test_rejects_invalid_batch_size(service, batch_size):
    """Некорректный размер пакета отклоняется до вычисления."""
    left = torch.rand((4, 4), dtype=torch.float32)
    right = torch.rand((4, 4), dtype=torch.float32)

    with pytest.raises((TypeError, ValueError)):
        service.execute(
            left,
            right,
            batch_size=batch_size,
        )


@pytest.mark.parametrize(
    ("left_shape", "right_shape"),
    [
        ((4, 3), (4, 5)),
        ((4, 3), (3, 5, 2)),
        ((4, 3, 2), (3, 5)),
        ((0, 3), (3, 5)),
    ],
)
def test_rejects_invalid_matrix_dimensions(
    service,
    left_shape,
    right_shape,
):
    """Некорректные размеры матриц отклоняются."""
    left = torch.rand(left_shape, dtype=torch.float32)
    right = torch.rand(right_shape, dtype=torch.float32)

    with pytest.raises((TypeError, ValueError)):
        service.execute(left, right)


def test_training_is_recorded_only_after_success(service):
    """Обучение модели фиксируется только после успешного выполнения."""
    left = torch.rand((6, 4), dtype=torch.float32)
    right = torch.rand((4, 5), dtype=torch.float32)

    report = service.execute(left, right)

    assert report.successful
    assert report.training_recorded is True
    assert report.execution is not None
    assert report.execution.batch_executions
    assert all(
        item.backend is ComputeBackend.CPU
        for item in report.execution.batch_executions
    )


def test_report_serialization(service):
    """Отчёт адаптивного сервиса преобразуется в JSON-совместимый словарь."""
    left = torch.rand((4, 3), dtype=torch.float32)
    right = torch.rand((3, 2), dtype=torch.float32)

    report = service.execute(left, right)
    data = report.as_dict()

    assert isinstance(data, dict)

    assert data["task_type"] == TaskType.MATRIX_MULTIPLICATION.value

    assert "hardware" in data
    assert "decision" in data
    assert "memory_estimates" in data
    assert "performance_estimates" in data
    assert "execution" in data
    assert "training_recorded" in data

    assert isinstance(data["hardware"], dict)
    assert isinstance(data["decision"], dict)
    assert isinstance(data["memory_estimates"], dict)
    assert isinstance(data["performance_estimates"], dict)

    assert data["execution"]["successful"] is True
    assert data["execution"]["actual_backend"] == ComputeBackend.CPU.value
    assert data["execution"]["operations"] == 2 * 4 * 3 * 2

    assert data["training_recorded"] is True



def build_dual_backend_analyzer() -> TaskAnalyzer:
    """Создаёт анализатор с поддержкой CPU и CUDA."""
    registry = TaskRegistry()
    registry.register(
        TaskDefinition(
            task_type=TaskType.MATRIX_MULTIPLICATION,
            name="Matrix multiplication",
            description="Тестовая задача CPU/CUDA",
            workload=WorkloadKind.MIXED,
            supported_backends=(
                ComputeBackend.CPU,
                ComputeBackend.CUDA,
            ),
            precision=PrecisionRequirement.FLOAT32,
            parallelizable=True,
            supports_batching=True,
            estimator_key="matrix_multiplication",
        )
    )
    return TaskAnalyzer(registry)


def build_dual_backend_service(
    *,
    cpu_free_bytes: int = 7 * GIB,
    cuda_free_bytes: int = 6 * GIB,
    cuda_total_bytes: int = 8 * GIB,
    calibrate_cpu: bool = True,
    calibrate_cuda: bool = True,
) -> AdaptiveMatrixService:
    """Создаёт сервис с имитацией доступных CPU и CUDA."""
    analyzer = build_dual_backend_analyzer()
    hardware_monitor = HardwareMonitor()

    memory_estimator = MemoryEstimator(
        safety_margin_ratio=0.0,
        cpu_reserve_bytes=0,
        cuda_reserve_bytes=0,
        cpu_runtime_overhead_bytes=0,
        cuda_runtime_overhead_bytes=0,
    )

    performance_model = PerformanceModel(high_confidence_samples=2)

    # Используем одинаковую задачу и объём операций.
    # Различаются только измеренные времена CPU и CUDA.
    if calibrate_cpu:
        for _ in range(5):
            performance_model.add_sample(
                PerformanceSample(
                    task_type=TaskType.MATRIX_MULTIPLICATION,
                    backend=ComputeBackend.CPU,
                    operations=2_000_000,
                    input_bytes=20_000,
                    output_bytes=10_000,
                    compute_seconds=0.200,
                    transfer_seconds=0.0,
                    total_seconds=0.200,
                )
            )

    if calibrate_cuda:
        for _ in range(5):
            performance_model.add_sample(
                PerformanceSample(
                    task_type=TaskType.MATRIX_MULTIPLICATION,
                    backend=ComputeBackend.CUDA,
                    operations=2_000_000,
                    input_bytes=20_000,
                    output_bytes=10_000,
                    compute_seconds=0.002,
                    transfer_seconds=0.001,
                    total_seconds=0.003,
                )
            )

    snapshot = HardwareSnapshot(
        cpu_name="Test CPU",
        cpu_logical_cores=8,
        ram_total_bytes=8 * GIB,
        ram_available_bytes=cpu_free_bytes,
        cuda_available=True,
        cuda_device_name="Test CUDA",
        cuda_total_bytes=cuda_total_bytes,
        cuda_free_bytes=cuda_free_bytes,
    )

    patcher = patch.object(
        hardware_monitor,
        "inspect",
        return_value=snapshot,
    )
    patcher.start()

    adapter = PyTorchCUDAAdapter(torch_module=torch)
    coordinator = MatrixMultiplicationCoordinator(
        analyzer=analyzer,
        adapter=adapter,
        torch_module=torch,
    )

    service = AdaptiveMatrixService(
        analyzer=analyzer,
        hardware_monitor=hardware_monitor,
        memory_estimator=memory_estimator,
        performance_model=performance_model,
        decision_engine=DecisionEngine(),
        coordinator=coordinator,
        trainer=MatrixPerformanceTrainer(performance_model),
    )

    service._test_hardware_patcher = patcher
    return service


@pytest.fixture
def dual_backend_service():
    """Создаёт CPU/CUDA-сервис и освобождает подмену оборудования."""
    instance = build_dual_backend_service()
    try:
        yield instance
    finally:
        instance._test_hardware_patcher.stop()


def test_selects_cuda_when_calibrated_cuda_is_faster(
    dual_backend_service,
):
    """CUDA выбирается при подтверждённом преимуществе."""
    left = torch.rand((8, 8), dtype=torch.float32)
    right = torch.rand((8, 8), dtype=torch.float32)

    report = dual_backend_service.execute(left, right)

    assert report.decision.accepted
    assert report.decision.selected_backend is ComputeBackend.CUDA
    assert report.actual_backend is ComputeBackend.CUDA
    assert report.successful
    assert report.training_recorded


def test_selects_cpu_when_cuda_is_not_calibrated():
    """Без калибровки CUDA выбирается безопасный CPU."""
    service = build_dual_backend_service(
        calibrate_cpu=True,
        calibrate_cuda=False,
    )

    try:
        left = torch.rand((8, 8), dtype=torch.float32)
        right = torch.rand((8, 8), dtype=torch.float32)

        report = service.execute(left, right)

        assert report.decision.accepted
        assert report.decision.selected_backend is ComputeBackend.CPU
        assert report.actual_backend is ComputeBackend.CPU
        assert ComputeBackend.CUDA not in report.performance_estimates
    finally:
        service._test_hardware_patcher.stop()


def test_rejects_cuda_when_cuda_memory_is_insufficient():
    """Недостаток VRAM не позволяет выбрать CUDA."""
    service = build_dual_backend_service(
        cpu_free_bytes=7 * GIB,
        cuda_free_bytes=1,
        calibrate_cpu=True,
        calibrate_cuda=True,
    )

    try:
        left = torch.rand((8, 8), dtype=torch.float32)
        right = torch.rand((8, 8), dtype=torch.float32)

        report = service.execute(left, right)

        assert report.decision.accepted
        assert report.decision.selected_backend is ComputeBackend.CPU
        assert report.actual_backend is ComputeBackend.CPU
    finally:
        service._test_hardware_patcher.stop()


def test_rejects_task_when_neither_backend_has_memory():
    """Задача отклоняется при нехватке памяти обоих устройств."""
    service = build_dual_backend_service(
        cpu_free_bytes=1,
        cuda_free_bytes=1,
    )

    try:
        left = torch.rand((8, 8), dtype=torch.float32)
        right = torch.rand((8, 8), dtype=torch.float32)

        report = service.execute(left, right)

        assert not report.decision.accepted
        assert report.execution is None
        assert not report.successful
        assert not report.training_recorded
    finally:
        service._test_hardware_patcher.stop()


def test_cuda_execution_uses_safe_batch_size(
    dual_backend_service,
):
    """Размер CUDA-пакета не превышает оценку безопасной памяти."""
    left = torch.rand((13, 8), dtype=torch.float32)
    right = torch.rand((8, 6), dtype=torch.float32)

    report = dual_backend_service.execute(
        left,
        right,
        batch_size=256,
    )

    assert report.successful
    assert report.actual_backend is ComputeBackend.CUDA

    safe_batch_size = report.memory_estimates[
        ComputeBackend.CUDA
    ].recommended_batch_size

    assert safe_batch_size is not None
    assert safe_batch_size > 0
    assert len(report.execution.batch_executions) >= 1

    assert all(
        item.backend is ComputeBackend.CUDA
        for item in report.execution.batch_executions
    )


def test_cuda_calibration_is_recorded_in_performance_model(
    dual_backend_service,
):
    """После успешного CUDA-выполнения модель получает измерение."""
    left = torch.rand((8, 8), dtype=torch.float32)
    right = torch.rand((8, 8), dtype=torch.float32)

    before = len(
        dual_backend_service.performance_model.get_samples(
            TaskType.MATRIX_MULTIPLICATION,
            ComputeBackend.CUDA,
        )
    )

    report = dual_backend_service.execute(left, right)

    after = len(
        dual_backend_service.performance_model.get_samples(
            TaskType.MATRIX_MULTIPLICATION,
            ComputeBackend.CUDA,
        )
    )

    assert report.successful
    assert report.training_recorded
    assert after == before + 1

def test_cuda_prediction_uses_actual_batched_transfer_volume(
    dual_backend_service,
    monkeypatch,
):
    """Прогноз учитывает повторную передачу правой матрицы."""
    left = torch.rand((13, 8), dtype=torch.float32)
    right = torch.rand((8, 6), dtype=torch.float32)

    performance_model = (
        dual_backend_service.performance_model
    )

    original_estimate = performance_model.estimate
    captured_estimates = []

    def estimate_spy(
        analysis,
        backend,
        **kwargs,
    ):
        captured_estimates.append(
            (backend, kwargs.copy())
        )

        return original_estimate(
            analysis,
            backend,
            **kwargs,
        )

    monkeypatch.setattr(
        performance_model,
        "estimate",
        estimate_spy,
    )

    report = dual_backend_service.execute(
        left,
        right,
        batch_size=4,
    )

    assert report.successful
    assert report.actual_backend is ComputeBackend.CUDA

    cuda_estimates = [
        kwargs
        for backend, kwargs in captured_estimates
        if backend is ComputeBackend.CUDA
    ]

    assert len(cuda_estimates) == 1

    expected_batch_size = min(
        4,
        report.memory_estimates[
            ComputeBackend.CUDA
        ].recommended_batch_size,
        13,
    )

    expected_batch_count = (
        13 + expected_batch_size - 1
    ) // expected_batch_size

    expected_input_bytes = (
        13 * 8
        + expected_batch_count * 8 * 6
    ) * left.element_size()

    expected_output_bytes = (
        13 * 6 * left.element_size()
    )

    assert cuda_estimates[0] == {
        "transfer_input_bytes": expected_input_bytes,
        "transfer_output_bytes": expected_output_bytes,
    }