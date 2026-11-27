"""Анализ точности прогнозов и качества решений планировщика LAIE."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any, Mapping, Sequence

from services.compute.experiments.experiment_models import (
    DecisionEvaluation,
    DecisionQuality,
    ExperimentMeasurement,
    MeasurementStatus,
    PredictionComparison,
)
from services.compute.scheduler.task_types import ComputeBackend


class ExperimentAnalyzer:
    """Сопоставляет прогнозы с независимыми фактическими измерениями."""

    _TIE_TOLERANCE = 1e-12

    @staticmethod
    def _backend(value: Any) -> ComputeBackend:
        """Преобразует значение backend в штатный enum."""
        if isinstance(value, ComputeBackend):
            return value

        if isinstance(value, str):
            try:
                return ComputeBackend(value)
            except ValueError as exc:
                raise ValueError(
                    f"Неизвестный backend: {value!r}"
                ) from exc

        raise TypeError(
            "backend должен быть ComputeBackend или строкой."
        )

    @staticmethod
    def _validate_records(
        records: Sequence[Mapping[str, Any]],
        *,
        name: str,
    ) -> None:
        """Проверяет структуру входных записей."""
        if not isinstance(records, (tuple, list)):
            raise TypeError(f"{name} должен быть tuple или list.")

        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise TypeError(
                    f"{name}[{index}] должен быть Mapping."
                )

    @staticmethod
    def _validate_measurements(
        measurements: Sequence[ExperimentMeasurement],
    ) -> None:
        """Проверяет типы фактических измерений."""
        if not isinstance(measurements, (tuple, list)):
            raise TypeError(
                "measurements должен быть tuple или list."
            )

        for index, measurement in enumerate(measurements):
            if not isinstance(measurement, ExperimentMeasurement):
                raise TypeError(
                    f"measurements[{index}] должен быть "
                    "ExperimentMeasurement."
                )

    @staticmethod
    def _validate_comparisons(
        comparisons: Sequence[PredictionComparison],
    ) -> None:
        """Проверяет типы сравнений прогнозов."""
        if not isinstance(comparisons, (tuple, list)):
            raise TypeError(
                "comparisons должен быть tuple или list."
            )

        for index, comparison in enumerate(comparisons):
            if not isinstance(comparison, PredictionComparison):
                raise TypeError(
                    f"comparisons[{index}] должен быть "
                    "PredictionComparison."
                )

    @staticmethod
    def _validate_case_id(record: Mapping[str, Any]) -> str:
        """Извлекает и проверяет идентификатор конфигурации."""
        case_id = record.get("case_id")

        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(
                "Каждая запись должна содержать непустой case_id."
            )

        return case_id

    @classmethod
    def _validate_time(
        cls,
        record: Mapping[str, Any],
        field: str,
        *,
        positive: bool = False,
    ) -> float:
        """Извлекает конечное неотрицательное время."""
        value = record.get(field)

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{field} должен быть числом.")

        result = float(value)

        if not math.isfinite(result) or result < 0:
            raise ValueError(
                f"{field} должен быть конечным неотрицательным числом."
            )

        if positive and result <= 0:
            raise ValueError(f"{field} должен быть больше нуля.")

        return result

    @staticmethod
    def _successful_measurements(
        measurements: Sequence[ExperimentMeasurement],
    ) -> dict[tuple[str, ComputeBackend], list[ExperimentMeasurement]]:
        """Группирует только успешные измерения по case_id и backend."""
        grouped: dict[
            tuple[str, ComputeBackend],
            list[ExperimentMeasurement],
        ] = defaultdict(list)

        for measurement in measurements:
            if measurement.status is not MeasurementStatus.SUCCESS:
                continue

            grouped[
                (measurement.case_id, measurement.actual_backend)
            ].append(measurement)

        return grouped

    def build_comparisons(
        self,
        predictions: Sequence[Mapping[str, Any]],
        measurements: Sequence[ExperimentMeasurement],
    ) -> tuple[PredictionComparison, ...]:
        """Строит сравнения прогнозов с фактическими измерениями.

        Используются только успешные измерения соответствующих
        case_id и backend. Для статистики требуется минимум два
        независимых измерения.
        """
        self._validate_records(predictions, name="predictions")
        self._validate_measurements(measurements)

        grouped = self._successful_measurements(measurements)
        comparisons: list[PredictionComparison] = []
        seen: set[tuple[str, ComputeBackend]] = set()

        for prediction in predictions:
            case_id = self._validate_case_id(prediction)
            backend = self._backend(prediction.get("backend"))
            key = (case_id, backend)

            if key in seen:
                raise ValueError(
                    f"Повторный прогноз для case_id={case_id!r}, "
                    f"backend={backend.value!r}."
                )

            seen.add(key)
            samples = grouped.get(key, [])

            if len(samples) < 2:
                continue

            predicted_compute = self._validate_time(
                prediction,
                "predicted_compute_seconds",
            )
            predicted_transfer = self._validate_time(
                prediction,
                "predicted_transfer_seconds",
            )
            predicted_total = self._validate_time(
                prediction,
                "predicted_total_seconds",
                positive=True,
            )

            compute_values = [
                float(item.compute_seconds)
                for item in samples
            ]
            transfer_values = [
                float(item.transfer_seconds)
                for item in samples
            ]
            total_values = [
                float(item.total_seconds)
                for item in samples
            ]

            comparisons.append(
                PredictionComparison(
                    case_id=case_id,
                    backend=backend,
                    predicted_compute_seconds=predicted_compute,
                    predicted_transfer_seconds=predicted_transfer,
                    predicted_total_seconds=predicted_total,
                    actual_compute_median=statistics.median(
                        compute_values
                    ),
                    actual_transfer_median=statistics.median(
                        transfer_values
                    ),
                    actual_total_median=statistics.median(
                        total_values
                    ),
                    actual_total_min=min(total_values),
                    actual_total_max=max(total_values),
                    actual_total_stddev=statistics.stdev(
                        total_values
                    ),
                    sample_count=len(samples),
                )
            )

        return tuple(comparisons)

    def build_prediction_errors(
        self,
        comparisons: Sequence[PredictionComparison],
    ) -> tuple[dict[str, Any], ...]:
        """Рассчитывает абсолютные, относительные и знаковые ошибки.

        Относительная ошибка рассчитывается относительно фактической
        медианы. Результат выражается долей единицы, а не процентами.
        """
        self._validate_comparisons(comparisons)

        results: list[dict[str, Any]] = []

        for comparison in comparisons:
            predicted = comparison.predicted_total_seconds
            actual = comparison.actual_total_median

            signed_error = predicted - actual
            absolute_error = abs(signed_error)
            relative_error = absolute_error / actual

            results.append(
                {
                    "case_id": comparison.case_id,
                    "backend": comparison.backend.value,
                    "predicted_total_seconds": predicted,
                    "actual_total_median": actual,
                    "signed_error_seconds": signed_error,
                    "absolute_error_seconds": absolute_error,
                    "relative_error_ratio": relative_error,
                    "relative_error_percent": relative_error * 100.0,
                    "sample_count": comparison.sample_count,
                }
            )

        return tuple(results)

    def evaluate_decisions(
        self,
        decisions: Sequence[Mapping[str, Any]],
        measurements: Sequence[ExperimentMeasurement],
    ) -> tuple[DecisionEvaluation, ...]:
        """Оценивает решения планировщика по фактическим медианам.

        Для сравнения CPU и CUDA необходимо минимум два успешных
        измерения каждого backend.
        """
        self._validate_records(decisions, name="decisions")
        self._validate_measurements(measurements)

        grouped = self._successful_measurements(measurements)

        medians: dict[
            str,
            dict[ComputeBackend, float],
        ] = defaultdict(dict)

        for (case_id, backend), samples in grouped.items():
            if len(samples) < 2:
                continue

            medians[case_id][backend] = statistics.median(
                float(item.total_seconds)
                for item in samples
            )

        results: list[DecisionEvaluation] = []
        seen_cases: set[str] = set()

        for decision in decisions:
            case_id = self._validate_case_id(decision)

            if case_id in seen_cases:
                raise ValueError(
                    f"Повторное решение для case_id={case_id!r}."
                )

            seen_cases.add(case_id)

            selected_value = decision.get("selected_backend")
            selected = (
                None
                if selected_value is None
                else self._backend(selected_value)
            )

            reason_code = decision.get("reason_code", "")
            explanation = decision.get("explanation", "")

            if not isinstance(reason_code, str):
                raise TypeError("reason_code должен быть строкой.")

            if not isinstance(explanation, str):
                raise TypeError("explanation должен быть строкой.")

            case_medians = medians.get(case_id, {})
            comparable = tuple(
                backend
                for backend in (ComputeBackend.CPU, ComputeBackend.CUDA)
                if backend in case_medians
            )

            if len(comparable) < 2:
                results.append(
                    DecisionEvaluation(
                        case_id=case_id,
                        selected_backend=selected,
                        empirical_fastest_backend=None,
                        quality=DecisionQuality.NOT_COMPARABLE,
                        regret_seconds=None,
                        regret_ratio=None,
                        reason_code=reason_code,
                        explanation=(
                            "Недостаточно независимых успешных измерений "
                            "для сравнения CPU и CUDA."
                        ),
                        comparable_backends=comparable,
                    )
                )
                continue

            fastest_time = min(
                case_medians[backend]
                for backend in comparable
            )

            fastest_backends = tuple(
                backend
                for backend in comparable
                if math.isclose(
                    case_medians[backend],
                    fastest_time,
                    rel_tol=0.0,
                    abs_tol=self._TIE_TOLERANCE,
                )
            )

            # Поле модели допускает только один backend.
            # При ничьей сохраняем детерминированный порядок CPU, CUDA.
            empirical_fastest = fastest_backends[0]
            tied = len(fastest_backends) > 1

            if selected is None:
                quality = DecisionQuality.INDETERMINATE
                regret_seconds = None
                regret_ratio = None
                result_explanation = (
                    "Планировщик не выбрал backend; "
                    "фактическое сравнение выполнено."
                )

            elif selected not in case_medians:
                quality = DecisionQuality.INDETERMINATE
                regret_seconds = None
                regret_ratio = None
                result_explanation = (
                    "Для выбранного backend недостаточно "
                    "успешных независимых измерений."
                )

            else:
                selected_time = case_medians[selected]
                regret_seconds = max(
                    0.0,
                    selected_time - fastest_time,
                )
                regret_ratio = regret_seconds / fastest_time

                if selected in fastest_backends:
                    quality = DecisionQuality.CORRECT
                    if tied:
                        result_explanation = (
                            "Ничья: выбранный backend статистически "
                            "не отличается по медианному времени "
                            "от другого самого быстрого backend "
                            "в пределах допуска."
                        )
                    else:
                        result_explanation = (
                            "Выбран фактически самый быстрый backend."
                        )
                else:
                    quality = DecisionQuality.SUBOPTIMAL
                    result_explanation = (
                        "Выбранный backend медленнее фактически "
                        "самого быстрого backend."
                    )

            results.append(
                DecisionEvaluation(
                    case_id=case_id,
                    selected_backend=selected,
                    empirical_fastest_backend=empirical_fastest,
                    quality=quality,
                    regret_seconds=regret_seconds,
                    regret_ratio=regret_ratio,
                    reason_code=reason_code,
                    explanation=result_explanation,
                    comparable_backends=comparable,
                )
            )

        return tuple(results)
