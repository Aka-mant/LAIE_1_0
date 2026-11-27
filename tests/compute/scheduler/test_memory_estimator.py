
"""Тесты оценщика потребления памяти."""

import json

import pytest

from services.compute.scheduler.memory_estimator import (
    MIB,
    MemoryDeviceSnapshot,
    MemoryEstimator,
)
from services.compute.scheduler.task_analyzer import TaskAnalysis
from services.compute.scheduler.task_types import (
    ComputeBackend,
    PrecisionRequirement,
    TaskType,
    WorkloadKind,
)


@pytest.fixture
def matrix_analysis():
    """Типовая задача матричного умножения."""
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


def make_snapshot(
    backend=ComputeBackend.CUDA,
    total_mib=8192,
    free_mib=7000,
):
    return MemoryDeviceSnapshot(
        backend=backend,
        total_bytes=total_mib * MIB,
        free_bytes=free_mib * MIB,
        device_name="Test device",
    )


def test_snapshot_accepts_valid_values():
    snapshot = make_snapshot()

    assert snapshot.backend == ComputeBackend.CUDA
    assert snapshot.total_bytes == 8192 * MIB
    assert snapshot.free_bytes == 7000 * MIB


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("total_bytes", True, TypeError),
        ("total_bytes", 1.5, TypeError),
        ("total_bytes", -1, ValueError),
        ("free_bytes", True, TypeError),
        ("free_bytes", -1, ValueError),
    ],
)
def test_snapshot_validates_memory_fields(field, value, error):
    values = {
        "backend": ComputeBackend.CUDA,
        "total_bytes": 8192 * MIB,
        "free_bytes": 7000 * MIB,
        "device_name": "Test device",
    }
    values[field] = value

    with pytest.raises(error):
        MemoryDeviceSnapshot(**values)


def test_snapshot_rejects_free_memory_above_total():
    with pytest.raises(ValueError, match="не может превышать"):
        make_snapshot(total_mib=100, free_mib=101)


def test_snapshot_rejects_invalid_backend():
    with pytest.raises(TypeError):
        MemoryDeviceSnapshot(
            backend="cuda",
            total_bytes=8192 * MIB,
            free_bytes=7000 * MIB,
        )


def test_snapshot_rejects_empty_device_name():
    with pytest.raises(ValueError, match="непустой"):
        MemoryDeviceSnapshot(
            backend=ComputeBackend.CUDA,
            total_bytes=8192 * MIB,
            free_bytes=7000 * MIB,
            device_name=" ",
        )


def test_estimator_uses_default_reserves():
    estimator = MemoryEstimator()

    assert estimator.cuda_reserve_bytes == 1536 * MIB
    assert estimator.cpu_reserve_bytes == 512 * MIB


@pytest.mark.parametrize(
    "ratio",
    [-0.1, 1.0, 2.0, float("inf"), float("nan"), True],
)
def test_estimator_rejects_invalid_safety_margin(ratio):
    with pytest.raises((TypeError, ValueError)):
        MemoryEstimator(safety_margin_ratio=ratio)


@pytest.mark.parametrize(
    "field",
    [
        "cpu_reserve_bytes",
        "cuda_reserve_bytes",
        "cpu_runtime_overhead_bytes",
        "cuda_runtime_overhead_bytes",
    ],
)
def test_estimator_rejects_negative_configuration(field):
    with pytest.raises(ValueError):
        MemoryEstimator(**{field: -1})


def test_estimator_rejects_boolean_configuration():
    with pytest.raises(TypeError):
        MemoryEstimator(cuda_reserve_bytes=True)


def test_cuda_estimate_includes_safety_and_runtime_overhead(
    estimator,
    matrix_analysis,
):
    result = estimator.estimate(
        matrix_analysis,
        make_snapshot(free_mib=7000),
    )

    assert result.estimated_payload_bytes == 10 * MIB
    assert result.safety_margin_bytes == 2 * MIB
    assert result.runtime_overhead_bytes == 256 * MIB
    assert result.required_bytes == 268 * MIB


def test_cuda_estimate_subtracts_reserved_memory(
    estimator,
    matrix_analysis,
):
    result = estimator.estimate(
        matrix_analysis,
        make_snapshot(free_mib=7000),
    )

    assert result.reserve_bytes == 1536 * MIB
    assert result.usable_bytes == 5464 * MIB
    assert result.fits is True


def test_task_fits_when_memory_is_sufficient(
    estimator,
    matrix_analysis,
):
    result = estimator.estimate(
        matrix_analysis,
        make_snapshot(free_mib=7000),
    )

    assert result.fits is True


def test_task_does_not_fit_when_memory_is_insufficient(
    estimator,
    matrix_analysis,
):
    result = estimator.estimate(
        matrix_analysis,
        make_snapshot(free_mib=1700),
    )

    assert result.fits is False
    assert result.usable_bytes == 164 * MIB


def test_task_does_not_fit_when_free_memory_is_below_reserve(
    estimator,
    matrix_analysis,
):
    result = estimator.estimate(
        matrix_analysis,
        make_snapshot(free_mib=1000),
    )

    assert result.usable_bytes == 0
    assert result.fits is False
    assert result.recommended_batch_size is None


def test_cpu_uses_cpu_reserve_and_overhead(
    estimator,
    matrix_analysis,
):
    result = estimator.estimate(
        matrix_analysis,
        make_snapshot(
            backend=ComputeBackend.CPU,
            total_mib=32768,
            free_mib=16000,
        ),
    )

    assert result.reserve_bytes == 512 * MIB
    assert result.runtime_overhead_bytes == 64 * MIB
    assert result.fits is True


def test_unsupported_backend_is_rejected(estimator):
    analysis = TaskAnalysis(
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

    with pytest.raises(ValueError, match="не поддерживает"):
        estimator.estimate(
            analysis,
            make_snapshot(backend=ComputeBackend.CUDA),
        )


def test_batch_recommendation_does_not_exceed_item_count(
    estimator,
    matrix_analysis,
):
    result = estimator.estimate(
        matrix_analysis,
        make_snapshot(free_mib=1800),
    )

    assert result.fits is False
    assert result.recommended_batch_size is not None
    assert 1 <= result.recommended_batch_size <= matrix_analysis.item_count
    assert result.batch_required_bytes <= result.usable_bytes


def test_batch_recommendation_is_none_when_one_item_does_not_fit(
    estimator,
    matrix_analysis,
):
    result = estimator.estimate(
        matrix_analysis,
        make_snapshot(free_mib=1792),
    )

    assert result.fits is False
    assert result.recommended_batch_size is None
    assert result.batch_required_bytes is None


def test_non_batchable_task_has_no_batch_recommendation(
    estimator,
    matrix_analysis,
):
    analysis = TaskAnalysis(
        task_type=matrix_analysis.task_type,
        item_count=matrix_analysis.item_count,
        estimated_operations=matrix_analysis.estimated_operations,
        estimated_input_bytes=matrix_analysis.estimated_input_bytes,
        estimated_output_bytes=matrix_analysis.estimated_output_bytes,
        estimated_workspace_bytes=matrix_analysis.estimated_workspace_bytes,
        estimated_memory_bytes=matrix_analysis.estimated_memory_bytes,
        workload=matrix_analysis.workload,
        supported_backends=matrix_analysis.supported_backends,
        precision=matrix_analysis.precision,
        parallelizable=matrix_analysis.parallelizable,
        supports_batching=False,
        complexity_model=matrix_analysis.complexity_model,
        assumptions=matrix_analysis.assumptions,
    )

    result = estimator.estimate(
        analysis,
        make_snapshot(free_mib=1700),
    )

    assert result.recommended_batch_size is None
    assert result.batch_required_bytes is None



def test_batch_size_increases_with_available_memory(
    estimator,
    matrix_analysis,
):
    low_memory = estimator.estimate(
        matrix_analysis,
        make_snapshot(free_mib=1800),
    )
    high_memory = estimator.estimate(
        matrix_analysis,
        make_snapshot(free_mib=1802),
    )

    assert low_memory.fits is False
    assert high_memory.fits is False

    assert low_memory.recommended_batch_size is not None
    assert high_memory.recommended_batch_size is not None

    assert (
        high_memory.recommended_batch_size
        > low_memory.recommended_batch_size
    )



def test_full_batch_is_recommended_when_task_fits(
    estimator,
    matrix_analysis,
):
    result = estimator.estimate(
        matrix_analysis,
        make_snapshot(free_mib=7000),
    )

    assert result.fits is True
    assert result.recommended_batch_size == matrix_analysis.item_count


def test_batch_required_memory_includes_overhead(
    estimator,
    matrix_analysis,
):
    result = estimator.estimate(
        matrix_analysis,
        make_snapshot(free_mib=1700),
    )

    if result.recommended_batch_size is not None:
        assert (
            result.batch_required_bytes
            >= result.runtime_overhead_bytes
        )


def test_estimate_contains_device_information(
    estimator,
    matrix_analysis,
):
    result = estimator.estimate(
        matrix_analysis,
        make_snapshot(),
    )

    assert result.device_name == "Test device"
    assert result.backend == ComputeBackend.CUDA
    assert result.task_type == TaskType.MATRIX_MULTIPLICATION


def test_estimate_exposes_mib_properties(
    estimator,
    matrix_analysis,
):
    result = estimator.estimate(
        matrix_analysis,
        make_snapshot(),
    )

    assert result.required_mib == pytest.approx(
        result.required_bytes / MIB
    )
    assert result.usable_mib == pytest.approx(
        result.usable_bytes / MIB
    )


def test_estimate_can_be_serialized_to_json(
    estimator,
    matrix_analysis,
):
    result = estimator.estimate(
        matrix_analysis,
        make_snapshot(),
    )

    serialized = json.dumps(result.as_dict(), ensure_ascii=False)

    assert "required_mib" in serialized
    assert "usable_mib" in serialized
    assert "Test device" in serialized
