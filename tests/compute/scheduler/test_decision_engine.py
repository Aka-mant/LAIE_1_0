
"""Тесты движка принятия вычислительных решений."""

import json

import pytest

from services.compute.scheduler.decision_engine import (
    DecisionEngine,
    DecisionReason,
    DecisionStatus,
)
from services.compute.scheduler.memory_estimator import (
    MIB,
    MemoryDeviceSnapshot,
    MemoryEstimator,
)
from services.compute.scheduler.performance_model import PerformanceEstimate
from services.compute.scheduler.task_analyzer import TaskAnalysis
from services.compute.scheduler.task_types import (
    ComputeBackend,
    PrecisionRequirement,
    TaskType,
    WorkloadKind,
)


@pytest.fixture
def analysis():
    """Задача, поддерживающая CPU и CUDA."""
    return TaskAnalysis(
        task_type=TaskType.MATRIX_MULTIPLICATION,
        item_count=1000,
        estimated_operations=2_000_000,
        estimated_input_bytes=4 * MIB,
        estimated_output_bytes=2 * MIB,
        estimated_workspace_bytes=4 * MIB,
        estimated_memory_bytes=10 * MIB,
        workload=WorkloadKind.GPU_PARALLEL,
        supported_backends=(
            ComputeBackend.CPU,
            ComputeBackend.CUDA,
        ),
        precision=PrecisionRequirement.FLOAT32,
        parallelizable=True,
        supports_batching=True,
        complexity_model="O(M*K*N)",
        assumptions=("Тестовая модель",),
    )


@pytest.fixture
def estimator():
    return MemoryEstimator(
        safety_margin_ratio=0.20,
        cpu_reserve_bytes=512 * MIB,
        cuda_reserve_bytes=1536 * MIB,
        cpu_runtime_overhead_bytes=64 * MIB,
        cuda_runtime_overhead_bytes=256 * MIB,
    )


@pytest.fixture
def engine():
    return DecisionEngine(minimum_gpu_speedup=0.10)


def make_performance(
    backend,
    total_seconds,
    *,
    confidence="high",
    calibrated=True,
):
    """Создаёт тестовый прогноз производительности."""
    return PerformanceEstimate(
        task_type=TaskType.MATRIX_MULTIPLICATION,
        backend=backend,
        predicted_compute_seconds=total_seconds * 0.8,
        predicted_transfer_seconds=(
            total_seconds * 0.1
            if backend == ComputeBackend.CUDA
            else 0.0
        ),
        predicted_overhead_seconds=(
            total_seconds * 0.1
            if backend == ComputeBackend.CUDA
            else total_seconds * 0.2
        ),
        predicted_total_seconds=total_seconds,
        sample_count=10,
        confidence=confidence,
        calibrated=calibrated,
    )


def make_memory(estimator, analysis, backend, free_mib=7000):
    """Создаёт оценку памяти для выбранного backend."""
    total_mib = 32768 if backend == ComputeBackend.CPU else 8192

    snapshot = MemoryDeviceSnapshot(
        backend=backend,
        total_bytes=total_mib * MIB,
        free_bytes=free_mib * MIB,
        device_name=f"Test {backend.value}",
    )

    return estimator.estimate(analysis, snapshot)


def make_memory_pair(estimator, analysis):
    return {
        ComputeBackend.CPU: make_memory(
            estimator,
            analysis,
            ComputeBackend.CPU,
        ),
        ComputeBackend.CUDA: make_memory(
            estimator,
            analysis,
            ComputeBackend.CUDA,
        ),
    }


def make_performance_pair(cpu_seconds=10.0, cuda_seconds=5.0):
    return {
        ComputeBackend.CPU: make_performance(
            ComputeBackend.CPU,
            cpu_seconds,
        ),
        ComputeBackend.CUDA: make_performance(
            ComputeBackend.CUDA,
            cuda_seconds,
        ),
    }


def test_engine_rejects_invalid_speedup():
    with pytest.raises(ValueError):
        DecisionEngine(minimum_gpu_speedup=1.0)


def test_engine_rejects_negative_speedup():
    with pytest.raises(ValueError):
        DecisionEngine(minimum_gpu_speedup=-0.1)


def test_engine_rejects_boolean_speedup():
    with pytest.raises(TypeError):
        DecisionEngine(minimum_gpu_speedup=True)


def test_engine_rejects_nan_speedup():
    with pytest.raises(ValueError):
        DecisionEngine(minimum_gpu_speedup=float("nan"))


def test_selects_cuda_when_gpu_is_significantly_faster(
    engine,
    estimator,
    analysis,
):
    result = engine.decide(
        analysis,
        make_memory_pair(estimator, analysis),
        make_performance_pair(10.0, 5.0),
    )

    assert result.status == DecisionStatus.SELECTED
    assert result.selected_backend == ComputeBackend.CUDA
    assert result.reason_code == DecisionReason.GPU_FASTER


def test_selects_cpu_when_cpu_is_faster(
    engine,
    estimator,
    analysis,
):
    result = engine.decide(
        analysis,
        make_memory_pair(estimator, analysis),
        make_performance_pair(5.0, 10.0),
    )

    assert result.selected_backend == ComputeBackend.CPU
    assert result.reason_code == DecisionReason.CPU_FASTER


def test_selects_cpu_when_gpu_advantage_is_below_threshold(
    engine,
    estimator,
    analysis,
):
    result = engine.decide(
        analysis,
        make_memory_pair(estimator, analysis),
        make_performance_pair(10.0, 9.5),
    )

    assert result.selected_backend == ComputeBackend.CPU
    assert result.reason_code == DecisionReason.GPU_ADVANTAGE_TOO_SMALL


def test_selects_cuda_at_exact_speedup_threshold(
    engine,
    estimator,
    analysis,
):
    result = engine.decide(
        analysis,
        make_memory_pair(estimator, analysis),
        make_performance_pair(10.0, 9.0),
    )

    assert result.selected_backend == ComputeBackend.CUDA
    assert result.reason_code == DecisionReason.GPU_FASTER


def test_selects_cpu_when_cuda_has_low_confidence(
    engine,
    estimator,
    analysis,
):
    memory = make_memory_pair(estimator, analysis)
    performance = make_performance_pair(10.0, 1.0)
    performance[ComputeBackend.CUDA] = make_performance(
        ComputeBackend.CUDA,
        1.0,
        confidence="low",
    )

    result = engine.decide(analysis, memory, performance)

    assert result.selected_backend == ComputeBackend.CPU
    assert result.reason_code == DecisionReason.GPU_NOT_CALIBRATED


def test_selects_cpu_when_cuda_is_not_calibrated(
    engine,
    estimator,
    analysis,
):
    memory = make_memory_pair(estimator, analysis)
    performance = make_performance_pair(10.0, 1.0)
    performance[ComputeBackend.CUDA] = make_performance(
        ComputeBackend.CUDA,
        1.0,
        calibrated=False,
    )

    result = engine.decide(analysis, memory, performance)

    assert result.selected_backend == ComputeBackend.CPU
    assert result.reason_code == DecisionReason.GPU_NOT_CALIBRATED


def test_selects_cpu_when_cpu_prediction_is_missing(
    engine,
    estimator,
    analysis,
):
    performance = {
        ComputeBackend.CUDA: make_performance(
            ComputeBackend.CUDA,
            1.0,
        )
    }

    result = engine.decide(
        analysis,
        make_memory_pair(estimator, analysis),
        performance,
    )

    assert result.selected_backend == ComputeBackend.CPU
    assert result.reason_code == DecisionReason.CPU_SAFE_FALLBACK


def test_selects_cpu_when_cpu_prediction_has_low_confidence(
    engine,
    estimator,
    analysis,
):
    performance = make_performance_pair(10.0, 1.0)
    performance[ComputeBackend.CPU] = make_performance(
        ComputeBackend.CPU,
        10.0,
        confidence="low",
    )

    result = engine.decide(
        analysis,
        make_memory_pair(estimator, analysis),
        performance,
    )

    assert result.selected_backend == ComputeBackend.CPU
    assert result.reason_code == DecisionReason.CPU_SAFE_FALLBACK


def test_cpu_is_selected_when_cuda_memory_is_insufficient(
    engine,
    estimator,
    analysis,
):
    memory = make_memory_pair(estimator, analysis)
    memory[ComputeBackend.CUDA] = make_memory(
        estimator,
        analysis,
        ComputeBackend.CUDA,
        free_mib=1000,
    )

    result = engine.decide(
        analysis,
        memory,
        make_performance_pair(10.0, 1.0),
    )

    assert result.selected_backend == ComputeBackend.CPU
    assert result.reason_code == DecisionReason.GPU_MEMORY_INSUFFICIENT


def test_cuda_is_selected_when_cpu_memory_is_insufficient(
    engine,
    estimator,
    analysis,
):
    memory = make_memory_pair(estimator, analysis)
    memory[ComputeBackend.CPU] = make_memory(
        estimator,
        analysis,
        ComputeBackend.CPU,
        free_mib=100,
    )

    result = engine.decide(
        analysis,
        memory,
        make_performance_pair(10.0, 5.0),
    )

    assert result.selected_backend == ComputeBackend.CUDA
    assert result.reason_code == DecisionReason.GPU_FASTER


def test_rejects_when_neither_device_has_enough_memory(
    engine,
    estimator,
    analysis,
):
    memory = {
        ComputeBackend.CPU: make_memory(
            estimator,
            analysis,
            ComputeBackend.CPU,
            free_mib=100,
        ),
        ComputeBackend.CUDA: make_memory(
            estimator,
            analysis,
            ComputeBackend.CUDA,
            free_mib=100,
        ),
    }

    result = engine.decide(
        analysis,
        memory,
        make_performance_pair(),
    )

    assert result.status == DecisionStatus.REJECTED
    assert result.selected_backend is None
    assert result.reason_code == DecisionReason.NO_DEVICE_FITS


def test_rejects_when_cpu_does_not_fit_and_cuda_is_uncalibrated(
    engine,
    estimator,
    analysis,
):
    memory = make_memory_pair(estimator, analysis)
    memory[ComputeBackend.CPU] = make_memory(
        estimator,
        analysis,
        ComputeBackend.CPU,
        free_mib=100,
    )

    performance = make_performance_pair()
    performance[ComputeBackend.CUDA] = make_performance(
        ComputeBackend.CUDA,
        1.0,
        confidence="low",
    )

    result = engine.decide(analysis, memory, performance)

    assert result.status == DecisionStatus.REJECTED
    assert result.reason_code == DecisionReason.GPU_NOT_CALIBRATED


def test_cpu_only_task_selects_cpu_without_performance_model(
    engine,
    estimator,
):
    cpu_analysis = TaskAnalysis(
        task_type=TaskType.COMBINATION_GENERATION,
        item_count=100,
        estimated_operations=1000,
        estimated_input_bytes=1024,
        estimated_output_bytes=2048,
        estimated_workspace_bytes=1024,
        estimated_memory_bytes=4096,
        workload=WorkloadKind.CPU_BOUND,
        supported_backends=(ComputeBackend.CPU,),
        precision=PrecisionRequirement.INTEGER,
        parallelizable=False,
        supports_batching=True,
        complexity_model="O(N)",
        assumptions=(),
    )

    memory = make_memory(
        estimator,
        cpu_analysis,
        ComputeBackend.CPU,
    )

    result = engine.decide(
        cpu_analysis,
        {ComputeBackend.CPU: memory},
        {},
    )

    assert result.selected_backend == ComputeBackend.CPU
    assert result.reason_code == DecisionReason.CPU_ONLY


def test_cuda_only_task_requires_calibrated_performance(
    engine,
    estimator,
):
    cuda_analysis = TaskAnalysis(
        task_type=TaskType.MODEL_TRAINING,
        item_count=100,
        estimated_operations=1000,
        estimated_input_bytes=1024,
        estimated_output_bytes=2048,
        estimated_workspace_bytes=1024,
        estimated_memory_bytes=4096,
        workload=WorkloadKind.GPU_PARALLEL,
        supported_backends=(ComputeBackend.CUDA,),
        precision=PrecisionRequirement.FLOAT32,
        parallelizable=True,
        supports_batching=True,
        complexity_model="O(N)",
        assumptions=(),
    )

    memory = make_memory(
        estimator,
        cuda_analysis,
        ComputeBackend.CUDA,
    )

    result = engine.decide(
        cuda_analysis,
        {ComputeBackend.CUDA: memory},
        {},
    )

    assert result.status == DecisionStatus.REJECTED
    assert result.reason_code == DecisionReason.GPU_NOT_CALIBRATED


def test_cuda_only_task_selects_cuda_with_valid_prediction(
    engine,
    estimator,
):
    cuda_analysis = TaskAnalysis(
        task_type=TaskType.MODEL_TRAINING,
        item_count=100,
        estimated_operations=1000,
        estimated_input_bytes=1024,
        estimated_output_bytes=2048,
        estimated_workspace_bytes=1024,
        estimated_memory_bytes=4096,
        workload=WorkloadKind.GPU_PARALLEL,
        supported_backends=(ComputeBackend.CUDA,),
        precision=PrecisionRequirement.FLOAT32,
        parallelizable=True,
        supports_batching=True,
        complexity_model="O(N)",
        assumptions=(),
    )

    memory = make_memory(
        estimator,
        cuda_analysis,
        ComputeBackend.CUDA,
    )

    performance = PerformanceEstimate(
        task_type=TaskType.MODEL_TRAINING,
        backend=ComputeBackend.CUDA,
        predicted_compute_seconds=1.0,
        predicted_transfer_seconds=0.1,
        predicted_overhead_seconds=0.1,
        predicted_total_seconds=1.2,
        sample_count=10,
        confidence="high",
        calibrated=True,
    )

    result = engine.decide(
        cuda_analysis,
        {ComputeBackend.CUDA: memory},
        {ComputeBackend.CUDA: performance},
    )

    assert result.selected_backend == ComputeBackend.CUDA


def test_batching_is_accepted_when_full_task_does_not_fit(
    engine,
    estimator,
    analysis,
):
    memory = make_memory_pair(estimator, analysis)

    # Для CUDA задача целиком не помещается, но один пакет допустим.
    memory[ComputeBackend.CUDA] = make_memory(
        estimator,
        analysis,
        ComputeBackend.CUDA,
        free_mib=1800,
    )

    result = engine.decide(
        analysis,
        memory,
        make_performance_pair(10.0, 5.0),
    )

    assert result.selected_backend == ComputeBackend.CUDA
    assert result.recommended_batch_size is not None
    assert result.recommended_batch_size < analysis.item_count


def test_result_contains_explanation_and_memory_details(
    engine,
    estimator,
    analysis,
):
    result = engine.decide(
        analysis,
        make_memory_pair(estimator, analysis),
        make_performance_pair(),
    )

    assert result.explanation
    assert result.memory_required_bytes is not None
    assert result.memory_usable_bytes is not None
    assert result.predicted_total_seconds is not None
    assert result.confidence == "high"


def test_result_is_json_serializable(
    engine,
    estimator,
    analysis,
):
    result = engine.decide(
        analysis,
        make_memory_pair(estimator, analysis),
        make_performance_pair(),
    )

    serialized = json.dumps(result.as_dict(), ensure_ascii=False)

    assert "selected_backend" in serialized
    assert "reason_code" in serialized
    assert "evaluated_backends" in serialized


def test_rejects_wrong_analysis_type(engine):
    with pytest.raises(TypeError):
        engine.decide({}, {}, {})


def test_rejects_wrong_memory_mapping(engine, analysis):
    with pytest.raises(TypeError):
        engine.decide(analysis, [], {})


def test_rejects_memory_estimate_for_wrong_backend(
    engine,
    estimator,
    analysis,
):
    memory = make_memory_pair(estimator, analysis)
    memory[ComputeBackend.CPU] = memory[ComputeBackend.CUDA]

    with pytest.raises(ValueError, match="не совпадает с ключом"):
        engine.decide(analysis, memory, {})


def test_rejects_memory_estimate_for_wrong_task(
    engine,
    estimator,
    analysis,
):
    memory = make_memory_pair(estimator, analysis)
    memory[ComputeBackend.CPU] = make_memory(
        estimator,
        analysis,
        ComputeBackend.CPU,
    )

    wrong_task = TaskAnalysis(
        task_type=TaskType.STATISTICAL_ANALYSIS,
        item_count=100,
        estimated_operations=1000,
        estimated_input_bytes=1024,
        estimated_output_bytes=1024,
        estimated_workspace_bytes=1024,
        estimated_memory_bytes=3072,
        workload=WorkloadKind.CPU_BOUND,
        supported_backends=(ComputeBackend.CPU,),
        precision=PrecisionRequirement.FLOAT64,
        parallelizable=False,
        supports_batching=True,
        complexity_model="O(N)",
        assumptions=(),
    )

    with pytest.raises(ValueError, match="Тип задачи"):
        engine.decide(wrong_task, memory, {})


def test_rejects_wrong_performance_estimate_type(
    engine,
    estimator,
    analysis,
):
    with pytest.raises(TypeError):
        engine.decide(
            analysis,
            make_memory_pair(estimator, analysis),
            {ComputeBackend.CPU: object()},
        )


def test_rejects_performance_estimate_for_wrong_backend(
    engine,
    estimator,
    analysis,
):
    performance = make_performance_pair()
    performance[ComputeBackend.CPU] = performance[ComputeBackend.CUDA]

    with pytest.raises(ValueError, match="не совпадает с ключом"):
        engine.decide(
            analysis,
            make_memory_pair(estimator, analysis),
            performance,
        )
