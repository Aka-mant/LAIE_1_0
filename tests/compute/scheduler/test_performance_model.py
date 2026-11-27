
"""Тесты модели прогнозирования производительности."""

import pytest

from services.compute.scheduler.performance_model import (
    PerformanceModel,
    PerformanceSample,
    UnsupportedBackendError,
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
def analyzer():
    return TaskAnalyzer(create_default_registry())


@pytest.fixture
def matrix_analysis(analyzer):
    return analyzer.analyze(
        TaskType.MATRIX_MULTIPLICATION,
        TaskParameters(
            matrix_m=100,
            matrix_k=100,
            matrix_n=100,
        ),
    )


def make_sample(
    backend=ComputeBackend.CPU,
    task_type=TaskType.MATRIX_MULTIPLICATION,
    operations=1_000,
    input_bytes=8_000,
    output_bytes=4_000,
    compute_seconds=0.01,
    transfer_seconds=0.0,
    total_seconds=0.01,
):
    return PerformanceSample(
        task_type=task_type,
        backend=backend,
        operations=operations,
        input_bytes=input_bytes,
        output_bytes=output_bytes,
        compute_seconds=compute_seconds,
        transfer_seconds=transfer_seconds,
        total_seconds=total_seconds,
    )


def test_model_starts_without_calibration():
    model = PerformanceModel()

    assert model.sample_count == 0
    assert model.calibration_summary() == []


def test_model_rejects_invalid_confidence_sample_count():
    with pytest.raises(ValueError):
        PerformanceModel(high_confidence_samples=1)


def test_model_rejects_boolean_confidence_sample_count():
    with pytest.raises(TypeError):
        PerformanceModel(high_confidence_samples=True)


def test_sample_rejects_invalid_task_type():
    with pytest.raises(TypeError):
        make_sample(task_type="matrix_multiplication")


def test_sample_rejects_invalid_backend():
    with pytest.raises(TypeError):
        make_sample(backend="cuda")


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("operations", 0, ValueError),
        ("operations", -1, ValueError),
        ("operations", True, TypeError),
        ("operations", 1.5, TypeError),
        ("input_bytes", -1, ValueError),
        ("input_bytes", True, TypeError),
        ("output_bytes", 1.5, TypeError),
    ],
)
def test_sample_validates_integer_fields(field, value, error):
    values = {
        "task_type": TaskType.MATRIX_MULTIPLICATION,
        "backend": ComputeBackend.CPU,
        "operations": 100,
        "input_bytes": 100,
        "output_bytes": 100,
        "compute_seconds": 0.1,
        "transfer_seconds": 0.0,
        "total_seconds": 0.1,
    }
    values[field] = value

    with pytest.raises(error):
        PerformanceSample(**values)


@pytest.mark.parametrize(
    "field,value",
    [
        ("compute_seconds", -0.1),
        ("transfer_seconds", -0.1),
        ("total_seconds", -0.1),
        ("compute_seconds", float("inf")),
        ("transfer_seconds", float("nan")),
    ],
)
def test_sample_rejects_invalid_timings(field, value):
    values = {
        "task_type": TaskType.MATRIX_MULTIPLICATION,
        "backend": ComputeBackend.CPU,
        "operations": 100,
        "input_bytes": 100,
        "output_bytes": 100,
        "compute_seconds": 0.1,
        "transfer_seconds": 0.0,
        "total_seconds": 0.1,
    }
    values[field] = value

    with pytest.raises(ValueError):
        PerformanceSample(**values)


def test_sample_requires_positive_compute_time():
    with pytest.raises(ValueError):
        make_sample(compute_seconds=0.0)


def test_sample_total_cannot_be_less_than_components():
    with pytest.raises(ValueError):
        make_sample(
            compute_seconds=0.1,
            transfer_seconds=0.2,
            total_seconds=0.25,
        )


def test_cpu_sample_cannot_have_transfer_time():
    with pytest.raises(ValueError):
        make_sample(
            backend=ComputeBackend.CPU,
            transfer_seconds=0.01,
            total_seconds=0.02,
        )


def test_add_sample_increases_sample_count():
    model = PerformanceModel()
    model.add_sample(make_sample())

    assert model.sample_count == 1


def test_add_sample_rejects_invalid_object():
    model = PerformanceModel()

    with pytest.raises(TypeError):
        model.add_sample({"operations": 100})


def test_samples_are_grouped_by_task_and_backend():
    model = PerformanceModel()
    model.add_sample(make_sample(backend=ComputeBackend.CPU))
    model.add_sample(
        make_sample(
            backend=ComputeBackend.CUDA,
            transfer_seconds=0.001,
            total_seconds=0.011,
        )
    )

    assert len(
        model.get_samples(
            TaskType.MATRIX_MULTIPLICATION,
            ComputeBackend.CPU,
        )
    ) == 1

    assert len(
        model.get_samples(
            TaskType.MATRIX_MULTIPLICATION,
            ComputeBackend.CUDA,
        )
    ) == 1


def test_get_samples_returns_immutable_tuple():
    model = PerformanceModel()
    model.add_sample(make_sample())

    samples = model.get_samples(
        TaskType.MATRIX_MULTIPLICATION,
        ComputeBackend.CPU,
    )

    assert isinstance(samples, tuple)


def test_estimate_returns_none_without_calibration(matrix_analysis):
    model = PerformanceModel()

    assert model.estimate(
        matrix_analysis,
        ComputeBackend.CPU,
    ) is None


def test_cpu_estimate_uses_measured_operations_rate(matrix_analysis):
    model = PerformanceModel()
    model.add_sample(
        make_sample(
            backend=ComputeBackend.CPU,
            operations=1_000,
            compute_seconds=0.01,
            total_seconds=0.01,
        )
    )

    estimate = model.estimate(
        matrix_analysis,
        ComputeBackend.CPU,
    )

    assert estimate is not None

    expected_compute = (
        0.01 / 1_000 * matrix_analysis.estimated_operations
    )

    assert estimate.predicted_compute_seconds == pytest.approx(
        expected_compute
    )
    assert estimate.predicted_transfer_seconds == 0.0
    assert estimate.predicted_overhead_seconds == 0.0
    assert estimate.predicted_total_seconds == pytest.approx(
        expected_compute
    )
    assert estimate.confidence == "low"


def test_cuda_estimate_includes_transfer_and_overhead(matrix_analysis):
    model = PerformanceModel()
    model.add_sample(
        make_sample(
            backend=ComputeBackend.CUDA,
            operations=1_000,
            input_bytes=8_000,
            output_bytes=4_000,
            compute_seconds=0.01,
            transfer_seconds=0.006,
            total_seconds=0.020,
        )
    )

    estimate = model.estimate(
        matrix_analysis,
        ComputeBackend.CUDA,
    )
    expected_compute = (
            0.01 / 1_000 * matrix_analysis.estimated_operations
    )
    expected_transfer = (
            0.006 / 12_000
            * (
                    matrix_analysis.estimated_input_bytes
                    + matrix_analysis.estimated_output_bytes
            )
    )

    assert estimate.predicted_compute_seconds == pytest.approx(
        expected_compute
    )
    assert estimate.predicted_transfer_seconds == pytest.approx(
        expected_transfer
    )
    assert estimate.predicted_overhead_seconds == pytest.approx(0.004)
    assert estimate.predicted_total_seconds == pytest.approx(
        expected_compute + expected_transfer + 0.004
    )



def test_estimate_uses_median_of_samples(matrix_analysis):
    model = PerformanceModel()

    for seconds in (0.01, 0.02, 0.10):
        model.add_sample(
            make_sample(
                backend=ComputeBackend.CPU,
                operations=1_000,
                compute_seconds=seconds,
                total_seconds=seconds,
            )
        )

    estimate = model.estimate(
        matrix_analysis,
        ComputeBackend.CPU,
    )

    assert estimate is not None
    assert estimate.predicted_compute_seconds == pytest.approx(
        0.02 / 1_000 * matrix_analysis.estimated_operations
    )
    assert estimate.sample_count == 3
    assert estimate.confidence == "medium"


def test_high_confidence_requires_configured_sample_count(matrix_analysis):
    model = PerformanceModel(high_confidence_samples=3)

    for _ in range(3):
        model.add_sample(make_sample())

    estimate = model.estimate(
        matrix_analysis,
        ComputeBackend.CPU,
    )

    assert estimate is not None
    assert estimate.confidence == "high"


def test_estimates_do_not_mix_task_types(analyzer):
    model = PerformanceModel()
    model.add_sample(make_sample())

    analysis = analyzer.analyze(
        TaskType.STATISTICAL_ANALYSIS,
        TaskParameters(
            sample_count=100,
            feature_count=10,
        ),
    )

    assert model.estimate(
        analysis,
        ComputeBackend.CPU,
    ) is None


def test_unsupported_backend_is_rejected(analyzer):
    model = PerformanceModel()

    analysis = analyzer.analyze(
        TaskType.COMBINATION_GENERATION,
        TaskParameters(
            candidate_count=100,
            item_width=5,
        ),
    )

    with pytest.raises(UnsupportedBackendError):
        model.estimate(analysis, ComputeBackend.CUDA)


def test_estimate_supported_returns_only_calibrated_backends(
    matrix_analysis,
):
    model = PerformanceModel()
    model.add_sample(
        make_sample(
            backend=ComputeBackend.CPU,
            compute_seconds=0.01,
            total_seconds=0.01,
        )
    )

    estimates = model.estimate_supported(matrix_analysis)

    assert len(estimates) == 1
    assert estimates[0].backend is ComputeBackend.CPU


def test_estimate_supported_sorts_by_predicted_time(matrix_analysis):
    model = PerformanceModel()

    model.add_sample(
        make_sample(
            backend=ComputeBackend.CPU,
            operations=1_000,
            compute_seconds=0.1,
            total_seconds=0.1,
        )
    )
    model.add_sample(
        make_sample(
            backend=ComputeBackend.CUDA,
            operations=1_000,
            compute_seconds=0.001,
            transfer_seconds=0.001,
            total_seconds=0.002,
        )
    )

    estimates = model.estimate_supported(matrix_analysis)

    assert len(estimates) == 2
    assert estimates[0].backend is ComputeBackend.CUDA
    assert estimates[1].backend is ComputeBackend.CPU


def test_calibration_summary_contains_group_information():
    model = PerformanceModel()
    model.add_sample(make_sample())
    model.add_sample(make_sample())

    summary = model.calibration_summary()

    assert len(summary) == 1
    assert summary[0]["task_type"] == "matrix_multiplication"
    assert summary[0]["backend"] == "cpu"
    assert summary[0]["sample_count"] == 2
    assert summary[0]["confidence"] == "medium"


def test_estimate_can_be_serialized(matrix_analysis):
    model = PerformanceModel()
    model.add_sample(make_sample())

    estimate = model.estimate(
        matrix_analysis,
        ComputeBackend.CPU,
    )

    assert estimate is not None
    data = estimate.as_dict()

    assert data["task_type"] == "matrix_multiplication"
    assert data["backend"] == "cpu"
    assert data["calibrated"] is True
    assert data["sample_count"] == 1


def test_cuda_estimate_accepts_actual_transfer_volume(matrix_analysis):
    model = PerformanceModel()
    model.add_sample(
        make_sample(
            backend=ComputeBackend.CUDA,
            operations=1_000,
            input_bytes=8_000,
            output_bytes=4_000,
            compute_seconds=0.01,
            transfer_seconds=0.006,
            total_seconds=0.020,
        )
    )

    estimate = model.estimate(
        matrix_analysis,
        ComputeBackend.CUDA,
        transfer_input_bytes=24_000,
        transfer_output_bytes=8_000,
    )

    expected_transfer = 0.006 / 12_000 * 32_000

    assert estimate is not None
    assert estimate.predicted_transfer_seconds == pytest.approx(
        expected_transfer
    )
    assert estimate.predicted_overhead_seconds == pytest.approx(0.004)
    assert estimate.predicted_total_seconds == pytest.approx(
        estimate.predicted_compute_seconds
        + expected_transfer
        + 0.004
    )


def test_cuda_estimate_without_transfer_override_preserves_old_behavior(
    matrix_analysis,
):
    model = PerformanceModel()
    model.add_sample(
        make_sample(
            backend=ComputeBackend.CUDA,
            operations=1_000,
            input_bytes=8_000,
            output_bytes=4_000,
            compute_seconds=0.01,
            transfer_seconds=0.006,
            total_seconds=0.020,
        )
    )

    estimate = model.estimate(
        matrix_analysis,
        ComputeBackend.CUDA,
    )

    expected_transfer = (
        0.006 / 12_000
        * (
            matrix_analysis.estimated_input_bytes
            + matrix_analysis.estimated_output_bytes
        )
    )

    assert estimate is not None
    assert estimate.predicted_transfer_seconds == pytest.approx(
        expected_transfer
    )


@pytest.mark.parametrize(
    "kwargs,error",
    [
        (
            {"transfer_input_bytes": 100},
            ValueError,
        ),
        (
            {"transfer_output_bytes": 100},
            ValueError,
        ),
        (
            {
                "transfer_input_bytes": True,
                "transfer_output_bytes": 100,
            },
            TypeError,
        ),
        (
            {
                "transfer_input_bytes": 100,
                "transfer_output_bytes": -1,
            },
            ValueError,
        ),
        (
            {
                "transfer_input_bytes": 1.5,
                "transfer_output_bytes": 100,
            },
            TypeError,
        ),
    ],
)
def test_cuda_estimate_validates_transfer_overrides(
    matrix_analysis,
    kwargs,
    error,
):
    model = PerformanceModel()
    model.add_sample(
        make_sample(
            backend=ComputeBackend.CUDA,
            transfer_seconds=0.001,
            total_seconds=0.012,
        )
    )

    with pytest.raises(error):
        model.estimate(
            matrix_analysis,
            ComputeBackend.CUDA,
            **kwargs,
        )


def test_cpu_estimate_rejects_nonzero_transfer_override(
    matrix_analysis,
):
    model = PerformanceModel()
    model.add_sample(make_sample(backend=ComputeBackend.CPU))

    with pytest.raises(ValueError):
        model.estimate(
            matrix_analysis,
            ComputeBackend.CPU,
            transfer_input_bytes=100,
            transfer_output_bytes=100,
        )
