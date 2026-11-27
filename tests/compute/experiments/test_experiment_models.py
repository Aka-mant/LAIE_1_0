
"""Тесты типизированных моделей эксперимента LAIE."""

import json
from dataclasses import replace

import pytest

from services.compute.experiments.experiment_models import (
    DecisionEvaluation,
    DecisionQuality,
    ExperimentCase,
    ExperimentConfig,
    ExperimentMeasurement,
    ExperimentResult,
    ExperimentSplit,
    ExperimentStatus,
    MeasurementStatus,
    PredictionComparison,
)
from services.compute.scheduler.task_types import ComputeBackend


@pytest.fixture
def config() -> ExperimentConfig:
    """Возвращает корректную конфигурацию."""
    return ExperimentConfig(
        calibration_sizes=(256, 512),
        validation_sizes=(768, 1024),
        batch_sizes=(64, 128),
        repeats=3,
        cuda_warmup=1,
        high_confidence_samples=2,
    )


@pytest.fixture
def case() -> ExperimentCase:
    """Возвращает корректную конфигурацию матричной задачи."""
    return ExperimentCase(
        split=ExperimentSplit.VALIDATION,
        matrix_m=256,
        matrix_k=256,
        matrix_n=256,
        batch_size=64,
        dtype="float64",
        seed=42,
    )


@pytest.fixture
def measurement() -> ExperimentMeasurement:
    """Возвращает успешное измерение CPU."""
    return ExperimentMeasurement(
        case_id="validation:256x256x256:batch=64:dtype=float64:seed=42",
        requested_backend=ComputeBackend.CPU,
        actual_backend=ComputeBackend.CPU,
        status=MeasurementStatus.SUCCESS,
        repeat_index=0,
        operations=2 * 256 * 256 * 256,
        input_bytes=1048576,
        output_bytes=524288,
        compute_seconds=0.01,
        transfer_seconds=0.0,
        total_seconds=0.01,
        peak_memory_bytes=1048576,
    )


@pytest.fixture
def comparison() -> PredictionComparison:
    """Возвращает корректное сравнение прогноза."""
    return PredictionComparison(
        case_id="validation:256x256x256:batch=64:dtype=float64:seed=42",
        backend=ComputeBackend.CPU,
        predicted_compute_seconds=0.009,
        predicted_transfer_seconds=0.0,
        predicted_total_seconds=0.01,
        actual_compute_median=0.01,
        actual_transfer_median=0.0,
        actual_total_median=0.01,
        actual_total_min=0.009,
        actual_total_max=0.011,
        actual_total_stddev=0.001,
        sample_count=3,
    )


@pytest.fixture
def decision() -> DecisionEvaluation:
    """Возвращает корректную оценку решения."""
    return DecisionEvaluation(
        case_id="validation:256x256x256:batch=64:dtype=float64:seed=42",
        selected_backend=ComputeBackend.CPU,
        empirical_fastest_backend=ComputeBackend.CPU,
        quality=DecisionQuality.CORRECT,
        regret_seconds=0.0,
        regret_ratio=0.0,
        reason_code="CPU_FASTER",
        explanation="CPU показал лучшее фактическое время.",
        comparable_backends=(ComputeBackend.CPU, ComputeBackend.CUDA),
    )


@pytest.fixture
def result(
    config: ExperimentConfig,
    case: ExperimentCase,
    measurement: ExperimentMeasurement,
    comparison: PredictionComparison,
    decision: DecisionEvaluation,
) -> ExperimentResult:
    """Возвращает полный корректный результат эксперимента."""
    return ExperimentResult(
        experiment_id="experiment_001",
        created_at="2026-11-25T16:00:00+00:00",
        status=ExperimentStatus.COMPLETED,
        config=config,
        hardware={"cpu": "Test CPU", "cuda_available": True},
        software={"python": "3.14"},
        cases=(case,),
        predictions=({"backend": "cpu", "total_seconds": 0.01},),
        measurements=(measurement,),
        comparisons=(comparison,),
        decisions=(decision,),
        calibration_summary=({"backend": "cpu", "sample_count": 5},),
        warnings=("Тестовое предупреждение",),
    )


# ExperimentConfig


def test_config_accepts_valid_values(config: ExperimentConfig) -> None:
    assert config.calibration_sizes == (256, 512)
    assert config.validation_sizes == (768, 1024)
    assert config.repeats == 3
    assert config.dtype == "float64"


def test_config_defaults_are_valid() -> None:
    config = ExperimentConfig()

    assert config.dtype == "float64"
    assert config.repeats == 10
    assert config.cuda_warmup == 3


@pytest.mark.parametrize(
    "field",
    ["calibration_sizes", "validation_sizes", "batch_sizes"],
)
def test_config_rejects_empty_size_tuples(field: str) -> None:
    with pytest.raises(ValueError, match="не может быть пустым"):
        ExperimentConfig(**{field: ()})


@pytest.mark.parametrize(
    "field",
    ["calibration_sizes", "validation_sizes", "batch_sizes"],
)
def test_config_rejects_non_tuple_sizes(field: str) -> None:
    with pytest.raises(TypeError, match="должен быть tuple"):
        ExperimentConfig(**{field: [256, 512]})


@pytest.mark.parametrize(
    "field",
    ["calibration_sizes", "validation_sizes", "batch_sizes"],
)
def test_config_rejects_non_integer_sizes(field: str) -> None:
    with pytest.raises(TypeError, match="только целые числа"):
        ExperimentConfig(**{field: (256, True)})


@pytest.mark.parametrize(
    "field",
    ["calibration_sizes", "validation_sizes", "batch_sizes"],
)
def test_config_rejects_non_positive_sizes(field: str) -> None:
    with pytest.raises(ValueError, match="положительные числа"):
        ExperimentConfig(**{field: (256, 0)})


@pytest.mark.parametrize(
    "field",
    ["calibration_sizes", "validation_sizes", "batch_sizes"],
)
def test_config_rejects_duplicate_sizes(field: str) -> None:
    with pytest.raises(ValueError, match="повторяющиеся"):
        ExperimentConfig(**{field: (256, 256)})


def test_config_rejects_intersecting_splits() -> None:
    with pytest.raises(ValueError, match="должны быть независимы"):
        ExperimentConfig(
            calibration_sizes=(256, 512),
            validation_sizes=(512, 1024),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repeats", 1),
        ("cuda_warmup", -1),
        ("high_confidence_samples", 1),
    ],
)
def test_config_rejects_invalid_limits(field: str, value: int) -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(**{field: value})


@pytest.mark.parametrize(
    "field",
    ["repeats", "cuda_warmup", "random_seed", "high_confidence_samples"],
)
def test_config_rejects_boolean_integer_fields(field: str) -> None:
    with pytest.raises(TypeError):
        ExperimentConfig(**{field: True})


@pytest.mark.parametrize("dtype", ["float32", "int32", "", 123])
def test_config_rejects_unsupported_dtype(dtype: object) -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(dtype=dtype)


@pytest.mark.parametrize("speedup", [-0.1, 1.0, 2.0, float("nan"), float("inf")])
def test_config_rejects_invalid_gpu_speedup(speedup: float) -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(minimum_gpu_speedup=speedup)


def test_config_rejects_boolean_gpu_speedup() -> None:
    with pytest.raises(TypeError):
        ExperimentConfig(minimum_gpu_speedup=True)


# ExperimentCase


def test_case_calculates_stable_id_and_operations(
    case: ExperimentCase,
) -> None:
    assert case.case_id == (
        "validation:256x256x256:batch=64:dtype=float64:seed=42"
    )
    assert case.operations == 2 * 256 * 256 * 256


def test_case_serializes_to_dict(case: ExperimentCase) -> None:
    payload = case.as_dict()

    assert payload["split"] == "validation"
    assert payload["dtype"] == "float64"
    assert payload["operations"] == 2 * 256 * 256 * 256


def test_case_rejects_invalid_split() -> None:
    with pytest.raises(TypeError, match="ExperimentSplit"):
        ExperimentCase(
            split="validation",
            matrix_m=256,
            matrix_k=256,
            matrix_n=256,
            batch_size=64,
            dtype="float64",
            seed=42,
        )


@pytest.mark.parametrize(
    "field",
    ["matrix_m", "matrix_k", "matrix_n", "batch_size"],
)
def test_case_rejects_non_positive_dimensions(
    field: str,
    case: ExperimentCase,
) -> None:
    with pytest.raises(ValueError, match="положительным"):
        replace(case, **{field: 0})


@pytest.mark.parametrize(
    "field",
    ["matrix_m", "matrix_k", "matrix_n", "batch_size", "seed"],
)
def test_case_rejects_boolean_integer_fields(
    field: str,
    case: ExperimentCase,
) -> None:
    with pytest.raises(TypeError, match="целым числом"):
        replace(case, **{field: True})


def test_case_rejects_unsupported_dtype(case: ExperimentCase) -> None:
    with pytest.raises(ValueError, match="float64"):
        replace(case, dtype="float32")


# ExperimentMeasurement


def test_measurement_accepts_valid_cpu_sample(
    measurement: ExperimentMeasurement,
) -> None:
    assert measurement.status is MeasurementStatus.SUCCESS
    assert measurement.actual_backend is ComputeBackend.CPU
    assert measurement.transfer_seconds == 0.0


def test_measurement_serializes_backend_values(
    measurement: ExperimentMeasurement,
) -> None:
    payload = measurement.as_dict()

    assert payload["requested_backend"] == "cpu"
    assert payload["actual_backend"] == "cpu"
    assert payload["status"] == "success"


def test_measurement_rejects_empty_case_id(
    measurement: ExperimentMeasurement,
) -> None:
    with pytest.raises(ValueError, match="case_id"):
        replace(measurement, case_id=" ")


@pytest.mark.parametrize(
    "field",
    ["repeat_index", "operations", "input_bytes", "output_bytes"],
)
def test_measurement_rejects_boolean_integer_fields(
    field: str,
    measurement: ExperimentMeasurement,
) -> None:
    with pytest.raises(TypeError, match="целым числом"):
        replace(measurement, **{field: True})


@pytest.mark.parametrize(
    "field",
    ["repeat_index", "input_bytes", "output_bytes"],
)
def test_measurement_rejects_negative_integer_fields(
    field: str,
    measurement: ExperimentMeasurement,
) -> None:
    with pytest.raises(ValueError, match="отрицательным"):
        replace(measurement, **{field: -1})


def test_measurement_rejects_zero_operations(
    measurement: ExperimentMeasurement,
) -> None:
    with pytest.raises(ValueError, match="больше нуля"):
        replace(measurement, operations=0)


@pytest.mark.parametrize(
    "field",
    ["compute_seconds", "transfer_seconds", "total_seconds"],
)
def test_measurement_rejects_non_finite_timings(
    field: str,
    measurement: ExperimentMeasurement,
) -> None:
    with pytest.raises(ValueError, match="конечным"):
        replace(measurement, **{field: float("nan")})


def test_measurement_rejects_negative_timing(
    measurement: ExperimentMeasurement,
) -> None:
    with pytest.raises(ValueError, match="неотрицательным"):
        replace(measurement, compute_seconds=-0.1)


def test_measurement_rejects_backend_mismatch(
    measurement: ExperimentMeasurement,
) -> None:
    with pytest.raises(ValueError, match="совпадения"):
        replace(measurement, actual_backend=ComputeBackend.CUDA)


def test_measurement_rejects_success_with_fallback(
    measurement: ExperimentMeasurement,
) -> None:
    with pytest.raises(ValueError, match="fallback"):
        replace(measurement, fallback_used=True)


def test_measurement_rejects_missing_success_timing(
    measurement: ExperimentMeasurement,
) -> None:
    with pytest.raises(ValueError, match="все три"):
        replace(measurement, total_seconds=None)


def test_measurement_rejects_zero_compute_time(
    measurement: ExperimentMeasurement,
) -> None:
    with pytest.raises(ValueError, match="больше нуля"):
        replace(measurement, compute_seconds=0.0)


def test_measurement_rejects_inconsistent_total_time(
    measurement: ExperimentMeasurement,
) -> None:
    with pytest.raises(ValueError, match="меньше суммы"):
        replace(measurement, total_seconds=0.005)


def test_measurement_accepts_fallback_record() -> None:
    item = ExperimentMeasurement(
        case_id="case",
        requested_backend=ComputeBackend.CUDA,
        actual_backend=ComputeBackend.CPU,
        status=MeasurementStatus.FALLBACK,
        repeat_index=0,
        operations=100,
        input_bytes=100,
        output_bytes=50,
        fallback_used=True,
        error="CUDA unavailable",
    )

    assert item.status is MeasurementStatus.FALLBACK
    assert item.actual_backend is ComputeBackend.CPU


def test_measurement_rejects_fallback_without_flag() -> None:
    with pytest.raises(ValueError, match="fallback_used=True"):
        ExperimentMeasurement(
            case_id="case",
            requested_backend=ComputeBackend.CUDA,
            actual_backend=ComputeBackend.CPU,
            status=MeasurementStatus.FALLBACK,
            repeat_index=0,
            operations=100,
            input_bytes=100,
            output_bytes=50,
        )


def test_measurement_rejects_actual_backend_for_failed_status() -> None:
    with pytest.raises(ValueError, match="не должны иметь actual_backend"):
        ExperimentMeasurement(
            case_id="case",
            requested_backend=ComputeBackend.CUDA,
            actual_backend=ComputeBackend.CUDA,
            status=MeasurementStatus.FAILED,
            repeat_index=0,
            operations=100,
            input_bytes=100,
            output_bytes=50,
        )


# PredictionComparison


def test_comparison_accepts_valid_values(
    comparison: PredictionComparison,
) -> None:
    assert comparison.sample_count == 3
    assert comparison.predicted_total_seconds == 0.01


def test_comparison_rejects_invalid_sample_count(
    comparison: PredictionComparison,
) -> None:
    with pytest.raises(ValueError, match="не меньше 2"):
        replace(comparison, sample_count=1)


def test_comparison_rejects_boolean_sample_count(
    comparison: PredictionComparison,
) -> None:
    with pytest.raises(TypeError, match="целым числом"):
        replace(comparison, sample_count=True)


def test_comparison_rejects_invalid_total_range(
    comparison: PredictionComparison,
) -> None:
    with pytest.raises(ValueError, match="меньше actual_total_min"):
        replace(comparison, actual_total_max=0.008)


def test_comparison_rejects_zero_predicted_total(
    comparison: PredictionComparison,
) -> None:
    with pytest.raises(ValueError, match="predicted_total_seconds"):
        replace(comparison, predicted_total_seconds=0.0)


# DecisionEvaluation


def test_decision_accepts_valid_values(
    decision: DecisionEvaluation,
) -> None:
    assert decision.quality is DecisionQuality.CORRECT
    assert decision.selected_backend is ComputeBackend.CPU


def test_decision_rejects_duplicate_comparable_backends(
    decision: DecisionEvaluation,
) -> None:
    with pytest.raises(ValueError, match="содержит дубли"):
        replace(
            decision,
            comparable_backends=(
                ComputeBackend.CPU,
                ComputeBackend.CPU,
            ),
        )


def test_decision_rejects_non_tuple_comparable_backends(
    decision: DecisionEvaluation,
) -> None:
    with pytest.raises(TypeError, match="должен быть tuple"):
        replace(
            decision,
            comparable_backends=[ComputeBackend.CPU],
        )


def test_decision_rejects_negative_regret(
    decision: DecisionEvaluation,
) -> None:
    with pytest.raises(ValueError, match="неотрицательным"):
        replace(decision, regret_seconds=-0.1)


def test_decision_accepts_missing_backend_and_regret() -> None:
    item = DecisionEvaluation(
        case_id="case",
        selected_backend=None,
        empirical_fastest_backend=None,
        quality=DecisionQuality.NOT_COMPARABLE,
        regret_seconds=None,
        regret_ratio=None,
        reason_code="NO_SUPPORTED_BACKEND",
        explanation="Нет доступных backend.",
        comparable_backends=(),
    )

    assert item.selected_backend is None
    assert item.quality is DecisionQuality.NOT_COMPARABLE


# ExperimentResult


def test_result_accepts_valid_components(
    result: ExperimentResult,
) -> None:
    assert result.status is ExperimentStatus.COMPLETED
    assert len(result.cases) == 1
    assert len(result.measurements) == 1


def test_result_rejects_empty_experiment_id(
    result: ExperimentResult,
) -> None:
    with pytest.raises(ValueError, match="experiment_id"):
        replace(result, experiment_id="")


def test_result_rejects_empty_created_at(
    result: ExperimentResult,
) -> None:
    with pytest.raises(ValueError, match="created_at"):
        replace(result, created_at="")


@pytest.mark.parametrize(
    "field",
    [
        "cases",
        "predictions",
        "measurements",
        "comparisons",
        "decisions",
        "calibration_summary",
        "warnings",
    ],
)
def test_result_requires_tuple_collections(
    field: str,
    result: ExperimentResult,
) -> None:
    with pytest.raises(TypeError, match="должен быть tuple"):
        replace(result, **{field: []})


def test_result_rejects_invalid_case_item(
    result: ExperimentResult,
) -> None:
    with pytest.raises(TypeError, match="ExperimentCase"):
        replace(result, cases=("invalid",))


def test_result_rejects_invalid_measurement_item(
    result: ExperimentResult,
) -> None:
    with pytest.raises(TypeError, match="ExperimentMeasurement"):
        replace(result, measurements=("invalid",))


def test_result_rejects_invalid_comparison_item(
    result: ExperimentResult,
) -> None:
    with pytest.raises(TypeError, match="PredictionComparison"):
        replace(result, comparisons=("invalid",))


def test_result_rejects_invalid_decision_item(
    result: ExperimentResult,
) -> None:
    with pytest.raises(TypeError, match="DecisionEvaluation"):
        replace(result, decisions=("invalid",))


def test_result_serializes_all_components_to_json(
    result: ExperimentResult,
) -> None:
    payload = result.as_dict()
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    decoded = json.loads(encoded)

    assert decoded["experiment_type"] == "prediction_accuracy"
    assert decoded["status"] == "completed"
    assert decoded["config"]["dtype"] == "float64"
    assert decoded["cases"][0]["split"] == "validation"
    assert decoded["cases"][0]["dtype"] == "float64"
    assert decoded["measurements"][0]["actual_backend"] == "cpu"
    assert decoded["comparisons"][0]["sample_count"] == 3
    assert decoded["decisions"][0]["quality"] == "correct"
    assert decoded["warnings"] == ["Тестовое предупреждение"]


def test_result_serialization_does_not_mutate_source(
    result: ExperimentResult,
) -> None:
    original_hardware = dict(result.hardware)
    original_software = dict(result.software)

    result.as_dict()

    assert dict(result.hardware) == original_hardware
    assert dict(result.software) == original_software
