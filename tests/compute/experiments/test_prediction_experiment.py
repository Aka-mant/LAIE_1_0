"""Проверки эквивалентности экспериментального и производственного расчётов."""

from unittest.mock import call, patch

import pytest

from services.compute.experiments.prediction_experiment import (
    PredictionExperimentCalculator,
)
from services.compute.scheduler.adaptive_matrix_service import (
    AdaptiveMatrixService,
)
from services.compute.scheduler.decision_engine import DecisionEngine
from services.compute.scheduler.execution_engine import ExecutionEngine
from services.compute.scheduler.hardware_monitor import HardwareMonitor
from services.compute.scheduler.matrix_coordinator import (
    MatrixMultiplicationCoordinator,
)
from services.compute.scheduler.matrix_performance_trainer import (
    MatrixPerformanceTrainer,
)
from services.compute.scheduler.memory_estimator import (
    MemoryDeviceSnapshot,
    MemoryEstimator,
)
from services.compute.scheduler.performance_model import PerformanceModel
from services.compute.scheduler.pytorch_adapter import PyTorchCUDAAdapter
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
def components():
    """Создаёт реальные компоненты без выполнения матричных вычислений."""
    analyzer = TaskAnalyzer(create_default_registry())
    memory_estimator = MemoryEstimator()
    performance_model = PerformanceModel(high_confidence_samples=5)
    decision_engine = DecisionEngine(minimum_gpu_speedup=0.10)

    adapter = PyTorchCUDAAdapter()
    coordinator = MatrixMultiplicationCoordinator(
        analyzer,
        adapter,
        execution_engine=ExecutionEngine(),
    )

    production = AdaptiveMatrixService(
        analyzer=analyzer,
        hardware_monitor=HardwareMonitor(),
        memory_estimator=memory_estimator,
        performance_model=performance_model,
        decision_engine=decision_engine,
        coordinator=coordinator,
        trainer=MatrixPerformanceTrainer(performance_model),
    )

    experiment = PredictionExperimentCalculator(
        analyzer=analyzer,
        memory_estimator=memory_estimator,
        performance_model=performance_model,
        decision_engine=decision_engine,
    )

    return {
        "analyzer": analyzer,
        "memory_estimator": memory_estimator,
        "performance_model": performance_model,
        "decision_engine": decision_engine,
        "production": production,
        "experiment": experiment,
    }


@pytest.fixture
def matrix_analysis(components):
    """Создаёт штатный анализ матриц с точностью float64."""
    return components["analyzer"].analyze(
        TaskType.MATRIX_MULTIPLICATION,
        TaskParameters(
            matrix_m=256,
            matrix_k=256,
            matrix_n=256,
        ),
    )


@pytest.fixture
def memory_snapshots():
    """Возвращает одинаковые контролируемые снимки памяти."""
    return (
        MemoryDeviceSnapshot(
            backend=ComputeBackend.CPU,
            total_bytes=32 * 1024**3,
            free_bytes=24 * 1024**3,
            device_name="TEST_CPU",
        ),
        MemoryDeviceSnapshot(
            backend=ComputeBackend.CUDA,
            total_bytes=8 * 1024**3,
            free_bytes=7 * 1024**3,
            device_name="TEST_CUDA",
        ),
    )


def test_analysis_uses_float64_and_matches_registry(matrix_analysis):
    """Анализатор сохраняет штатную точность матричной задачи."""
    assert matrix_analysis.task_type is TaskType.MATRIX_MULTIPLICATION
    assert matrix_analysis.precision.value == "float64"
    assert matrix_analysis.estimated_operations == 2 * 256**3
    assert matrix_analysis.estimated_input_bytes == 2 * 256**2 * 8
    assert matrix_analysis.estimated_output_bytes == 256**2 * 8


def test_memory_estimates_match_adaptive_matrix_service(
    components,
    matrix_analysis,
    memory_snapshots,
):
    """Экспериментальные оценки памяти совпадают с производственными."""
    production = components["production"]
    experiment = components["experiment"]

    production_estimates = production._build_memory_estimates(
        matrix_analysis,
        memory_snapshots,
        rows=256,
        inner=256,
        columns=256,
        element_size=8,
    )

    experiment_estimates = experiment.build_memory_estimates(
        matrix_analysis,
        memory_snapshots,
        rows=256,
        inner=256,
        columns=256,
        element_size=8,
    )

    assert experiment_estimates == production_estimates

    assert (
        experiment_estimates[ComputeBackend.CPU].recommended_batch_size
        is None
    )

    assert (
        experiment_estimates[ComputeBackend.CUDA].recommended_batch_size
        == production_estimates[ComputeBackend.CUDA].recommended_batch_size
    )


def test_cuda_batch_search_matches_production(
    components,
    matrix_analysis,
    memory_snapshots,
):
    """Бинарный поиск CUDA-пакета совпадает с производственным."""
    production = components["production"]
    experiment = components["experiment"]

    cuda_snapshot = next(
        snapshot
        for snapshot in memory_snapshots
        if snapshot.backend is ComputeBackend.CUDA
    )

    expected = production._estimate_cuda_batch(
        matrix_analysis,
        cuda_snapshot,
        rows=256,
        inner=256,
        columns=256,
        element_size=8,
    )

    actual = experiment.estimate_cuda_batch(
        matrix_analysis,
        cuda_snapshot,
        rows=256,
        inner=256,
        columns=256,
        element_size=8,
    )

    assert actual == expected


def test_performance_estimates_use_identical_transfer_volumes(
    components,
    matrix_analysis,
    memory_snapshots,
):
    """Оба калькулятора передают модели одинаковые объёмы данных."""
    production = components["production"]
    experiment = components["experiment"]
    performance_model = components["performance_model"]

    memory_estimates = experiment.build_memory_estimates(
        matrix_analysis,
        memory_snapshots,
        rows=256,
        inner=256,
        columns=256,
        element_size=8,
    )

    expected_batch_size = min(
        64,
        memory_estimates[ComputeBackend.CUDA].recommended_batch_size,
        256,
    )
    batch_count = (256 + expected_batch_size - 1) // expected_batch_size

    expected_transfer_input = (
        256 * 256 + batch_count * 256 * 256
    ) * 8
    expected_transfer_output = 256 * 256 * 8

    with patch.object(
        performance_model,
        "estimate",
        wraps=performance_model.estimate,
    ) as estimate_mock:
        production._build_performance_estimates(
            matrix_analysis,
            memory_estimates,
            rows=256,
            inner=256,
            columns=256,
            element_size=8,
            requested_batch_size=64,
        )

        production_calls = estimate_mock.call_args_list

        estimate_mock.reset_mock()

        experiment.build_performance_estimates(
            matrix_analysis,
            memory_estimates,
            rows=256,
            inner=256,
            columns=256,
            element_size=8,
            requested_batch_size=64,
        )

        experiment_calls = estimate_mock.call_args_list

    assert experiment_calls == production_calls

    assert experiment_calls == [
        call(matrix_analysis, ComputeBackend.CPU),
        call(
            matrix_analysis,
            ComputeBackend.CUDA,
            transfer_input_bytes=expected_transfer_input,
            transfer_output_bytes=expected_transfer_output,
        ),
    ]


def test_decision_matches_adaptive_matrix_service(
    components,
    matrix_analysis,
    memory_snapshots,
):
    """DecisionEngine получает одинаковые входные данные и решение."""
    production = components["production"]
    experiment = components["experiment"]

    memory_estimates = experiment.build_memory_estimates(
        matrix_analysis,
        memory_snapshots,
        rows=256,
        inner=256,
        columns=256,
        element_size=8,
    )

    performance_estimates = experiment.build_performance_estimates(
        matrix_analysis,
        memory_estimates,
        rows=256,
        inner=256,
        columns=256,
        element_size=8,
        requested_batch_size=64,
    )

    production_decision = production.decision_engine.decide(
        matrix_analysis,
        memory_estimates,
        performance_estimates,
    )

    experiment_decision = experiment.decide(
        matrix_analysis,
        memory_estimates,
        performance_estimates,
    )

    assert experiment_decision == production_decision


def test_untrained_model_produces_no_performance_estimates(
    components,
    matrix_analysis,
    memory_snapshots,
):
    """Без реальных калибровочных данных прогнозы отсутствуют."""
    experiment = components["experiment"]

    memory_estimates = experiment.build_memory_estimates(
        matrix_analysis,
        memory_snapshots,
        rows=256,
        inner=256,
        columns=256,
        element_size=8,
    )

    estimates = experiment.build_performance_estimates(
        matrix_analysis,
        memory_estimates,
        rows=256,
        inner=256,
        columns=256,
        element_size=8,
        requested_batch_size=64,
    )

    assert estimates == {}


def test_frozen_model_is_independent(components):
    """Замороженная модель не является исходным объектом."""
    source = components["performance_model"]

    frozen = PredictionExperimentCalculator.clone_frozen_model(source)

    assert frozen is not source
    assert frozen.high_confidence_samples == source.high_confidence_samples

    for backend in (ComputeBackend.CPU, ComputeBackend.CUDA):
        assert frozen.get_samples(
            TaskType.MATRIX_MULTIPLICATION,
            backend,
        ) == source.get_samples(
            TaskType.MATRIX_MULTIPLICATION,
            backend,
        )


@pytest.mark.parametrize(
    ("rows", "inner", "columns", "element_size"),
    [
        (0, 256, 256, 8),
        (256, 0, 256, 8),
        (256, 256, 0, 8),
        (256, 256, 256, 0),
        (True, 256, 256, 8),
    ],
)
def test_invalid_dimensions_are_rejected(
    components,
    rows,
    inner,
    columns,
    element_size,
):
    """Некорректные размерности не допускаются в расчёты."""
    with pytest.raises((TypeError, ValueError)):
        components["experiment"].build_memory_analysis(
            components["analyzer"].analyze(
                TaskType.MATRIX_MULTIPLICATION,
                TaskParameters(
                    matrix_m=256,
                    matrix_k=256,
                    matrix_n=256,
                ),
            ),
            backend=ComputeBackend.CPU,
            rows=rows,
            inner=inner,
            columns=columns,
            element_size=element_size,
        )


def test_invalid_requested_batch_size_is_rejected(
    components,
    matrix_analysis,
    memory_snapshots,
):
    """Нулевой размер пакета не допускается."""
    experiment = components["experiment"]

    memory_estimates = experiment.build_memory_estimates(
        matrix_analysis,
        memory_snapshots,
        rows=256,
        inner=256,
        columns=256,
        element_size=8,
    )

    with pytest.raises(ValueError):
        experiment.build_performance_estimates(
            matrix_analysis,
            memory_estimates,
            rows=256,
            inner=256,
            columns=256,
            element_size=8,
            requested_batch_size=0,
        )
