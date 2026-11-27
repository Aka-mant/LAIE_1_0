"""Тесты анализа точности прогнозов и качества решений LAIE."""

import math

import pytest

from services.compute.experiments.experiment_analyzer import (
    ExperimentAnalyzer,
)
from services.compute.experiments.experiment_models import (
    DecisionQuality,
    ExperimentMeasurement,
    MeasurementStatus,
)
from services.compute.scheduler.task_types import ComputeBackend


@pytest.fixture
def analyzer():
    return ExperimentAnalyzer()


def make_measurement(
    case_id,
    backend,
    total,
    *,
    compute=None,
    transfer=0.0,
    status=MeasurementStatus.SUCCESS,
    repeat_index=0,
):
    """Создаёт изолированное тестовое измерение."""
    if compute is None:
        compute = total - transfer

    actual_backend = (
        backend
        if status in (
            MeasurementStatus.SUCCESS,
            MeasurementStatus.FALLBACK,
        )
        else None
    )

    fallback = status is MeasurementStatus.FALLBACK

    return ExperimentMeasurement(
        case_id=case_id,
        requested_backend=(
            ComputeBackend.CUDA
            if fallback
            else backend
        ),
        actual_backend=actual_backend,
        status=status,
        repeat_index=repeat_index,
        operations=1000,
        input_bytes=8000,
        output_bytes=4000,
        compute_seconds=(
            compute
            if status is MeasurementStatus.SUCCESS
            else None
        ),
        transfer_seconds=(
            transfer
            if status is MeasurementStatus.SUCCESS
            else None
        ),
        total_seconds=(
            total
            if status is MeasurementStatus.SUCCESS
            else None
        ),
        fallback_used=fallback,
        error="test fallback" if fallback else None,
    )


def make_prediction(
    case_id="validation:10x10x10:batch=10:dtype=float64:seed=42",
    backend=ComputeBackend.CPU,
    *,
    compute=1.0,
    transfer=0.0,
    total=1.0,
):
    return {
        "case_id": case_id,
        "backend": backend.value,
        "predicted_compute_seconds": compute,
        "predicted_transfer_seconds": transfer,
        "predicted_total_seconds": total,
    }


def make_decision(
    case_id,
    selected_backend,
    *,
    reason_code="GPU_FASTER",
):
    return {
        "case_id": case_id,
        "selected_backend": (
            selected_backend.value
            if selected_backend is not None
            else None
        ),
        "reason_code": reason_code,
        "explanation": "Тестовое решение",
    }


def test_build_comparisons_calculates_statistics(analyzer):
    """Сравнение рассчитывает медианы, диапазон и стандартное отклонение."""
    case_id = "case-1"

    measurements = [
        make_measurement(case_id, ComputeBackend.CPU, 1.0, compute=0.9),
        make_measurement(case_id, ComputeBackend.CPU, 2.0, compute=1.8),
        make_measurement(case_id, ComputeBackend.CPU, 3.0, compute=2.7),
    ]

    result = analyzer.build_comparisons(
        [make_prediction(case_id, total=2.0)],
        measurements,
    )

    assert len(result) == 1

    comparison = result[0]

    assert comparison.backend is ComputeBackend.CPU
    assert comparison.actual_compute_median == pytest.approx(1.8)
    assert comparison.actual_transfer_median == 0.0
    assert comparison.actual_total_median == 2.0
    assert comparison.actual_total_min == 1.0
    assert comparison.actual_total_max == 3.0
    assert comparison.actual_total_stddev == pytest.approx(1.0)
    assert comparison.sample_count == 3


def test_build_comparisons_groups_backends_independently(analyzer):
    """CPU и CUDA не смешиваются при расчёте статистики."""
    case_id = "case-2"

    measurements = [
        make_measurement(case_id, ComputeBackend.CPU, 2.0),
        make_measurement(case_id, ComputeBackend.CPU, 2.2),
        make_measurement(
            case_id,
            ComputeBackend.CUDA,
            1.0,
            compute=0.7,
            transfer=0.3,
        ),
        make_measurement(
            case_id,
            ComputeBackend.CUDA,
            1.2,
            compute=0.9,
            transfer=0.3,
        ),
    ]

    predictions = [
        make_prediction(case_id, ComputeBackend.CPU, total=2.1),
        make_prediction(
            case_id,
            ComputeBackend.CUDA,
            compute=0.8,
            transfer=0.3,
            total=1.1,
        ),
    ]

    result = analyzer.build_comparisons(predictions, measurements)

    assert len(result) == 2
    assert result[0].backend is ComputeBackend.CPU
    assert result[0].actual_total_median == pytest.approx(2.1)
    assert result[1].backend is ComputeBackend.CUDA
    assert result[1].actual_total_median == pytest.approx(1.1)
    assert result[1].actual_transfer_median == pytest.approx(0.3)


def test_build_comparisons_ignores_failed_skipped_and_fallback(analyzer):
    """Неуспешные измерения и fallback не участвуют в статистике."""
    case_id = "case-3"

    measurements = [
        make_measurement(case_id, ComputeBackend.CPU, 1.0),
        make_measurement(case_id, ComputeBackend.CPU, 1.2),
        make_measurement(
            case_id,
            ComputeBackend.CPU,
            0.1,
            status=MeasurementStatus.FAILED,
        ),
        make_measurement(
            case_id,
            ComputeBackend.CPU,
            0.1,
            status=MeasurementStatus.SKIPPED,
        ),
        make_measurement(
            case_id,
            ComputeBackend.CPU,
            0.1,
            status=MeasurementStatus.FALLBACK,
        ),
    ]

    result = analyzer.build_comparisons(
        [make_prediction(case_id, total=1.1)],
        measurements,
    )

    assert len(result) == 1
    assert result[0].sample_count == 2
    assert result[0].actual_total_median == pytest.approx(1.1)


def test_build_comparisons_skips_single_measurement(analyzer):
    """Одного измерения недостаточно для статистического сравнения."""
    case_id = "case-4"

    result = analyzer.build_comparisons(
        [make_prediction(case_id)],
        [make_measurement(case_id, ComputeBackend.CPU, 1.0)],
    )

    assert result == ()


def test_build_comparisons_rejects_duplicate_predictions(analyzer):
    """Повторный прогноз одного backend для конфигурации запрещён."""
    case_id = "case-5"
    prediction = make_prediction(case_id)

    with pytest.raises(ValueError, match="Повторный прогноз"):
        analyzer.build_comparisons(
            [prediction, prediction],
            [],
        )


def test_build_comparisons_rejects_unknown_backend(analyzer):
    """Неизвестный backend не принимается."""
    prediction = make_prediction()
    prediction["backend"] = "unknown"

    with pytest.raises(ValueError, match="Неизвестный backend"):
        analyzer.build_comparisons([prediction], [])


def test_build_comparisons_rejects_invalid_prediction_time(analyzer):
    """Отрицательное прогнозное время запрещено."""
    prediction = make_prediction()
    prediction["predicted_total_seconds"] = -1.0

    measurements = [
        make_measurement(prediction["case_id"], ComputeBackend.CPU, 1.0),
        make_measurement(prediction["case_id"], ComputeBackend.CPU, 1.2),
    ]

    with pytest.raises(ValueError, match="неотрицательным"):
        analyzer.build_comparisons([prediction], measurements)


def test_build_prediction_errors_calculates_all_error_metrics(analyzer):
    """Ошибка прогноза рассчитывается относительно фактической медианы."""
    case_id = "case-6"

    measurements = [
        make_measurement(case_id, ComputeBackend.CPU, 2.0),
        make_measurement(case_id, ComputeBackend.CPU, 2.0),
    ]

    comparisons = analyzer.build_comparisons(
        [make_prediction(case_id, total=2.5)],
        measurements,
    )

    errors = analyzer.build_prediction_errors(comparisons)

    assert len(errors) == 1

    error = errors[0]

    assert error["signed_error_seconds"] == pytest.approx(0.5)
    assert error["absolute_error_seconds"] == pytest.approx(0.5)
    assert error["relative_error_ratio"] == pytest.approx(0.25)
    assert error["relative_error_percent"] == pytest.approx(25.0)


def test_build_prediction_errors_detects_underestimation(analyzer):
    """Отрицательная знаковая ошибка означает занижение прогноза."""
    case_id = "case-7"

    measurements = [
        make_measurement(case_id, ComputeBackend.CPU, 2.0),
        make_measurement(case_id, ComputeBackend.CPU, 2.0),
    ]

    comparisons = analyzer.build_comparisons(
        [make_prediction(case_id, total=1.5)],
        measurements,
    )

    error = analyzer.build_prediction_errors(comparisons)[0]

    assert error["signed_error_seconds"] == pytest.approx(-0.5)
    assert error["absolute_error_seconds"] == pytest.approx(0.5)


def test_build_prediction_errors_zero_error(analyzer):
    """Совпадение прогноза с фактической медианой даёт нулевую ошибку."""
    case_id = "case-8"

    measurements = [
        make_measurement(case_id, ComputeBackend.CPU, 2.0),
        make_measurement(case_id, ComputeBackend.CPU, 2.0),
    ]

    comparisons = analyzer.build_comparisons(
        [make_prediction(case_id, total=2.0)],
        measurements,
    )

    error = analyzer.build_prediction_errors(comparisons)[0]

    assert error["signed_error_seconds"] == 0.0
    assert error["absolute_error_seconds"] == 0.0
    assert error["relative_error_ratio"] == 0.0


def test_build_prediction_errors_rejects_invalid_input(analyzer):
    """Метод ошибок принимает только PredictionComparison."""
    with pytest.raises(TypeError, match="PredictionComparison"):
        analyzer.build_prediction_errors([{}])


def test_evaluate_decisions_marks_cpu_correct(analyzer):
    """Выбор CPU правильный, если его фактическая медиана минимальна."""
    case_id = "case-9"

    measurements = [
        make_measurement(case_id, ComputeBackend.CPU, 1.0),
        make_measurement(case_id, ComputeBackend.CPU, 1.2),
        make_measurement(case_id, ComputeBackend.CUDA, 2.0),
        make_measurement(case_id, ComputeBackend.CUDA, 2.2),
    ]

    result = analyzer.evaluate_decisions(
        [make_decision(case_id, ComputeBackend.CPU)],
        measurements,
    )

    evaluation = result[0]

    assert evaluation.quality is DecisionQuality.CORRECT
    assert evaluation.empirical_fastest_backend is ComputeBackend.CPU
    assert evaluation.regret_seconds == 0.0
    assert evaluation.regret_ratio == 0.0


def test_evaluate_decisions_marks_cuda_correct(analyzer):
    """Выбор CUDA правильный, если его фактическая медиана минимальна."""
    case_id = "case-10"

    measurements = [
        make_measurement(case_id, ComputeBackend.CPU, 2.0),
        make_measurement(case_id, ComputeBackend.CPU, 2.2),
        make_measurement(case_id, ComputeBackend.CUDA, 1.0),
        make_measurement(case_id, ComputeBackend.CUDA, 1.2),
    ]

    evaluation = analyzer.evaluate_decisions(
        [make_decision(case_id, ComputeBackend.CUDA)],
        measurements,
    )[0]

    assert evaluation.quality is DecisionQuality.CORRECT
    assert evaluation.empirical_fastest_backend is ComputeBackend.CUDA
    assert evaluation.regret_seconds == 0.0


def test_evaluate_decisions_calculates_regret(analyzer):
    """Субоптимальный выбор получает абсолютный и относительный regret."""
    case_id = "case-11"

    measurements = [
        make_measurement(case_id, ComputeBackend.CPU, 2.0),
        make_measurement(case_id, ComputeBackend.CPU, 2.0),
        make_measurement(case_id, ComputeBackend.CUDA, 1.0),
        make_measurement(case_id, ComputeBackend.CUDA, 1.0),
    ]

    evaluation = analyzer.evaluate_decisions(
        [make_decision(case_id, ComputeBackend.CPU)],
        measurements,
    )[0]

    assert evaluation.quality is DecisionQuality.SUBOPTIMAL
    assert evaluation.empirical_fastest_backend is ComputeBackend.CUDA
    assert evaluation.regret_seconds == pytest.approx(1.0)
    assert evaluation.regret_ratio == pytest.approx(1.0)


def test_evaluate_decisions_tie_accepts_both_backends(analyzer):
    """При равных медианах оба backend считаются правильным выбором."""
    case_id = "case-12"

    measurements = [
        make_measurement(case_id, ComputeBackend.CPU, 1.0),
        make_measurement(case_id, ComputeBackend.CPU, 1.0),
        make_measurement(case_id, ComputeBackend.CUDA, 1.0),
        make_measurement(case_id, ComputeBackend.CUDA, 1.0),
    ]

    for backend in (ComputeBackend.CPU, ComputeBackend.CUDA):
        evaluation = analyzer.evaluate_decisions(
            [make_decision(case_id, backend)],
            measurements,
        )[0]

        assert evaluation.quality is DecisionQuality.CORRECT
        assert evaluation.regret_seconds == 0.0
        assert "Ничья" in evaluation.explanation


def test_evaluate_decisions_not_comparable_with_one_backend(analyzer):
    """Без двух измерений каждого backend сравнение невозможно."""
    case_id = "case-13"

    measurements = [
        make_measurement(case_id, ComputeBackend.CPU, 1.0),
        make_measurement(case_id, ComputeBackend.CPU, 1.2),
    ]

    evaluation = analyzer.evaluate_decisions(
        [make_decision(case_id, ComputeBackend.CPU)],
        measurements,
    )[0]

    assert evaluation.quality is DecisionQuality.NOT_COMPARABLE
    assert evaluation.empirical_fastest_backend is None
    assert evaluation.regret_seconds is None
    assert evaluation.regret_ratio is None


def test_evaluate_decisions_ignores_fallback_measurements(analyzer):
    """Fallback не может влиять на оценку CPU/CUDA."""
    case_id = "case-14"

    measurements = [
        make_measurement(case_id, ComputeBackend.CPU, 1.0),
        make_measurement(case_id, ComputeBackend.CPU, 1.2),
        make_measurement(
            case_id,
            ComputeBackend.CPU,
            0.1,
            status=MeasurementStatus.FALLBACK,
        ),
        make_measurement(case_id, ComputeBackend.CUDA, 2.0),
    ]

    evaluation = analyzer.evaluate_decisions(
        [make_decision(case_id, ComputeBackend.CPU)],
        measurements,
    )[0]

    assert evaluation.quality is DecisionQuality.NOT_COMPARABLE


def test_evaluate_decisions_without_selection_is_indeterminate(analyzer):
    """Отсутствие выбранного backend не маскируется как ошибка измерений."""
    case_id = "case-15"

    measurements = [
        make_measurement(case_id, ComputeBackend.CPU, 1.0),
        make_measurement(case_id, ComputeBackend.CPU, 1.0),
        make_measurement(case_id, ComputeBackend.CUDA, 2.0),
        make_measurement(case_id, ComputeBackend.CUDA, 2.0),
    ]

    evaluation = analyzer.evaluate_decisions(
        [make_decision(case_id, None)],
        measurements,
    )[0]

    assert evaluation.quality is DecisionQuality.INDETERMINATE
    assert evaluation.empirical_fastest_backend is ComputeBackend.CPU
    assert evaluation.regret_seconds is None


def test_evaluate_decisions_rejects_duplicate_cases(analyzer):
    """Повторная оценка одной конфигурации запрещена."""
    case_id = "case-16"
    decision = make_decision(case_id, ComputeBackend.CPU)

    with pytest.raises(ValueError, match="Повторное решение"):
        analyzer.evaluate_decisions([decision, decision], [])


def test_validate_records_rejects_invalid_container(analyzer):
    """Входные записи должны передаваться списком или кортежем."""
    with pytest.raises(TypeError, match="tuple или list"):
        analyzer.build_comparisons(
            iter(()),
            [],
        )


def test_validate_measurements_rejects_invalid_item(analyzer):
    """Измерения должны иметь тип ExperimentMeasurement."""
    with pytest.raises(TypeError, match="ExperimentMeasurement"):
        analyzer.build_comparisons(
            [],
            [{}],
        )


def test_relative_error_is_finite(analyzer):
    """Относительная ошибка остаётся конечной для положительного факта."""
    case_id = "case-17"

    measurements = [
        make_measurement(case_id, ComputeBackend.CPU, 0.001),
        make_measurement(case_id, ComputeBackend.CPU, 0.001),
    ]

    comparisons = analyzer.build_comparisons(
        [make_prediction(case_id, total=0.002)],
        measurements,
    )

    error = analyzer.build_prediction_errors(comparisons)[0]

    assert math.isfinite(error["relative_error_ratio"])
    assert error["relative_error_ratio"] == pytest.approx(1.0)
