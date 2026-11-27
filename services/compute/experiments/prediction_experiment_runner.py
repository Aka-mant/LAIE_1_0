"""Изолированный исполнитель эксперимента точности прогнозирования LAIE.

Калибровка использует только проверенные реальные MatrixExecutionReport.
Validation выполняется независимой замороженной моделью и никогда
не передаёт измерения в MatrixPerformanceTrainer.
"""

from __future__ import annotations

import hashlib
import importlib
from math import isfinite
import platform
import sys
import uuid
from datetime import datetime, timezone
from typing import Any

from services.compute.experiments.experiment_analyzer import (
    ExperimentAnalyzer,
)
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
)
from services.compute.experiments.experiment_storage import (
    ExperimentStorage,
)
from services.compute.experiments.prediction_experiment import (
    PredictionExperimentCalculator,
)
from services.compute.scheduler.execution_engine import ExecutionEngine
from services.compute.scheduler.hardware_monitor import HardwareMonitor
from services.compute.scheduler.matrix_coordinator import (
    MatrixExecutionReport,
    MatrixMultiplicationCoordinator,
)
from services.compute.scheduler.matrix_performance_trainer import (
    MatrixPerformanceTrainer,
)
from services.compute.scheduler.performance_model import PerformanceModel
from services.compute.scheduler.pytorch_adapter import PyTorchCUDAAdapter
from services.compute.scheduler.task_analyzer import TaskAnalyzer
from services.compute.scheduler.task_registry import create_default_registry
from services.compute.scheduler.task_types import (
    ComputeBackend,
    TaskType,
)


class PredictionExperimentRunner:
    """Выполняет полный цикл калибровки и независимой проверки прогнозов."""

    def __init__(
        self,
        config: ExperimentConfig | None = None,
        *,
        torch_module: Any | None = None,
        storage: ExperimentStorage | None = None,
        hardware_monitor: HardwareMonitor | None = None,
        performance_model: PerformanceModel | None = None,
    ) -> None:
        if config is not None and not isinstance(config, ExperimentConfig):
            raise TypeError("config должен быть ExperimentConfig.")

        if storage is not None and not isinstance(storage, ExperimentStorage):
            raise TypeError("storage должен быть ExperimentStorage.")

        if hardware_monitor is not None and not isinstance(
            hardware_monitor,
            HardwareMonitor,
        ):
            raise TypeError("hardware_monitor должен быть HardwareMonitor.")

        if performance_model is not None and not isinstance(
            performance_model,
            PerformanceModel,
        ):
            raise TypeError("performance_model должен быть PerformanceModel.")

        self.config = config or ExperimentConfig()

        self.torch = torch_module or importlib.import_module("torch")

        self.storage = storage or ExperimentStorage()
        self.hardware_monitor = hardware_monitor or HardwareMonitor(self.torch)

        model_settings = (
            performance_model
            if performance_model is not None
            else PerformanceModel(
                high_confidence_samples=self.config.high_confidence_samples,
            )
        )

        if (
            model_settings.high_confidence_samples
            != self.config.high_confidence_samples
        ):
            raise ValueError(
                "high_confidence_samples модели не совпадает с настройкой эксперимента."
            )

        # Эксперимент не изменяет переданную извне или production-модель.
        self.performance_model = PerformanceModel(
            high_confidence_samples=self.config.high_confidence_samples,
        )

        self.trainer = MatrixPerformanceTrainer(self.performance_model)

        self.task_analyzer = TaskAnalyzer(create_default_registry())

        self.calculator = PredictionExperimentCalculator.create_default(
            performance_model=self.performance_model,
            minimum_gpu_speedup=self.config.minimum_gpu_speedup,
            high_confidence_samples=self.config.high_confidence_samples,
        )

        self.adapter = PyTorchCUDAAdapter(self.torch)

        # В эксперименте автоматический CPU fallback запрещён.
        self.coordinator = MatrixMultiplicationCoordinator(
            analyzer=self.task_analyzer,
            adapter=self.adapter,
            execution_engine=ExecutionEngine(
                cpu_fallback_enabled=False,
            ),
            torch_module=self.torch,
        )

        self.analyzer = ExperimentAnalyzer()
        self._has_run = False

    def run(self) -> ExperimentResult:
        """Запускает эксперимент и атомарно сохраняет его результат."""
        if self._has_run:
            raise RuntimeError(
                "Экземпляр PredictionExperimentRunner можно запускать только один раз."
            )

        self._has_run = True

        experiment_id = uuid.uuid4().hex
        created_at = datetime.now(timezone.utc).isoformat()

        cases = self._build_cases()
        measurements: list[ExperimentMeasurement] = []
        predictions: list[dict[str, Any]] = []
        decisions: list[DecisionEvaluation] = []
        warnings: list[str] = []

        hardware_data: dict[str, Any] = {}
        software_data = self._software_metadata()

        status = ExperimentStatus.RUNNING
        error: str | None = None
        comparisons = ()

        try:
            hardware_snapshot = self.hardware_monitor.inspect()
            hardware_data = hardware_snapshot.as_dict()
            memory_snapshots = hardware_snapshot.memory_snapshots()

            cuda_available = hardware_snapshot.cuda_available

            # Этап 1. Калибровка.
            calibration_cases = [
                case for case in cases if case.split is ExperimentSplit.CALIBRATION
            ]

            for case in calibration_cases:
                left, right = self._create_matrices(case)

                self._warmup_cuda(
                    case,
                    left,
                    right,
                    cuda_available=cuda_available,
                    warnings=warnings,
                )

                for backend in (
                    ComputeBackend.CPU,
                    ComputeBackend.CUDA,
                ):
                    if backend is ComputeBackend.CUDA and not cuda_available:
                        measurements.extend(
                            self._skipped_measurements(
                                case,
                                backend,
                            )
                        )
                        continue

                    for repeat_index in range(self.config.repeats):
                        measurement, report = self._measure(
                            case,
                            left,
                            right,
                            backend,
                            repeat_index,
                        )
                        measurements.append(measurement)

                        if report is None:
                            continue

                        if not self._is_calibration_eligible(
                            case,
                            report,
                            backend,
                        ):
                            continue

                        try:
                            accepted = self.trainer.record(report)
                        except Exception as exc:
                            warnings.append(
                                "Ошибка обучения на отчёте "
                                f"{case.case_id}, {backend.value}, "
                                f"repeat={repeat_index}: "
                                f"{type(exc).__name__}: {exc}"
                            )
                            continue

                        if not accepted:
                            warnings.append(
                                "Trainer отклонил успешный отчёт: "
                                f"{case.case_id}, {backend.value}, "
                                f"repeat={repeat_index}."
                            )

            # После калибровки создаётся независимая frozen-модель.
            frozen_model = self.calculator.clone_frozen_model(
                self.performance_model,
            )

            frozen_calculator = PredictionExperimentCalculator.create_default(
                performance_model=frozen_model,
                minimum_gpu_speedup=self.config.minimum_gpu_speedup,
                high_confidence_samples=self.config.high_confidence_samples,
            )

            # Этап 2. Прогнозы для validation создаются до измерений.
            validation_cases = [
                case for case in cases if case.split is ExperimentSplit.VALIDATION
            ]

            decision_records: list[dict[str, Any]] = []

            for case in validation_cases:
                analysis = self.task_analyzer.analyze(
                    TaskType.MATRIX_MULTIPLICATION,
                    self._task_parameters(case),
                )

                memory_estimates = frozen_calculator.build_memory_estimates(
                    analysis,
                    memory_snapshots,
                    rows=case.matrix_m,
                    inner=case.matrix_k,
                    columns=case.matrix_n,
                )

                performance_estimates = frozen_calculator.build_performance_estimates(
                    analysis,
                    memory_estimates,
                    rows=case.matrix_m,
                    inner=case.matrix_k,
                    columns=case.matrix_n,
                    requested_batch_size=case.batch_size,
                )

                decision = frozen_calculator.decide(
                    analysis,
                    memory_estimates,
                    performance_estimates,
                )

                reason_code = decision.reason_code.value
                selected_backend = decision.selected_backend

                reliable_backends: list[ComputeBackend] = []

                for backend, estimate in performance_estimates.items():
                    prediction = {
                        "case_id": case.case_id,
                        "backend": backend.value,
                        "predicted_compute_seconds": (
                            estimate.predicted_compute_seconds
                        ),
                        "predicted_transfer_seconds": (
                            estimate.predicted_transfer_seconds
                        ),
                        "predicted_total_seconds": (estimate.predicted_total_seconds),
                        "sample_count": estimate.sample_count,
                        "confidence": estimate.confidence,
                        "calibrated": estimate.calibrated,
                        "selected_backend": (
                            selected_backend.value
                            if selected_backend is not None
                            else None
                        ),
                        "reason_code": reason_code,
                    }

                    predictions.append(prediction)

                    if (
                        estimate.calibrated
                        and estimate.confidence in {"medium", "high"}
                        and estimate.predicted_total_seconds > 0
                    ):
                        reliable_backends.append(backend)

                decision_records.append(
                    {
                        "case_id": case.case_id,
                        "selected_backend": (
                            selected_backend.value
                            if selected_backend is not None
                            else None
                        ),
                        "reason_code": reason_code,
                        "explanation": decision.explanation,
                        "reliable_backends": tuple(reliable_backends),
                    }
                )

            # Этап 3. Независимые фактические validation-измерения.
            for case in validation_cases:
                left, right = self._create_matrices(case)

                self._warmup_cuda(
                    case,
                    left,
                    right,
                    cuda_available=cuda_available,
                    warnings=warnings,
                )

                for backend in (
                    ComputeBackend.CPU,
                    ComputeBackend.CUDA,
                ):
                    if backend is ComputeBackend.CUDA and not cuda_available:
                        measurements.extend(
                            self._skipped_measurements(
                                case,
                                backend,
                            )
                        )
                        continue

                    for repeat_index in range(self.config.repeats):
                        measurement, _ = self._measure(
                            case,
                            left,
                            right,
                            backend,
                            repeat_index,
                        )

                        # Критически важно: validation не обучает trainer.
                        measurements.append(measurement)

            # Этап 4. Сравнение прогнозов с независимыми измерениями.
            validation_ids = {case.case_id for case in validation_cases}

            validation_measurements = tuple(
                item for item in measurements if item.case_id in validation_ids
            )

            comparisons = self.analyzer.build_comparisons(
                predictions,
                validation_measurements,
            )

            prediction_errors = self.analyzer.build_prediction_errors(comparisons)
            errors_by_key = {
                (item["case_id"], item["backend"]): item for item in prediction_errors
            }

            for prediction in predictions:
                key = (
                    prediction["case_id"],
                    prediction["backend"],
                )
                prediction_error = errors_by_key.get(key)

                if prediction_error is not None:
                    prediction["prediction_error"] = prediction_error

            # Этап 5. Оценка решений с обязательным gate калибровки.
            for record in decision_records:
                reliable_backends = record["reliable_backends"]

                if (
                    ComputeBackend.CPU not in reliable_backends
                    or ComputeBackend.CUDA not in reliable_backends
                ):
                    decisions.append(
                        DecisionEvaluation(
                            case_id=record["case_id"],
                            selected_backend=(
                                ComputeBackend(record["selected_backend"])
                                if record["selected_backend"] is not None
                                else None
                            ),
                            empirical_fastest_backend=None,
                            quality=DecisionQuality.NOT_COMPARABLE,
                            regret_seconds=None,
                            regret_ratio=None,
                            reason_code=record["reason_code"],
                            explanation=(
                                "Решение не оценивается: для обоих "
                                "backend отсутствуют надёжные "
                                "калиброванные прогнозы."
                            ),
                            comparable_backends=(),
                        )
                    )
                    continue

                evaluated = self.analyzer.evaluate_decisions(
                    (
                        {
                            "case_id": record["case_id"],
                            "selected_backend": record["selected_backend"],
                            "reason_code": record["reason_code"],
                            "explanation": record["explanation"],
                        },
                    ),
                    validation_measurements,
                )

                decisions.extend(evaluated)

            # Определение итогового состояния.
            successful_count = sum(
                item.status is MeasurementStatus.SUCCESS for item in measurements
            )

            failed_count = sum(
                item.status
                in (
                    MeasurementStatus.FAILED,
                    MeasurementStatus.FALLBACK,
                )
                for item in measurements
            )

            skipped_count = sum(
                item.status is MeasurementStatus.SKIPPED for item in measurements
            )

            if successful_count == 0:
                status = ExperimentStatus.FAILED
                error = "Эксперимент не получил ни одного успешного измерения."
            elif failed_count or skipped_count or warnings:
                status = ExperimentStatus.PARTIAL
            else:
                status = ExperimentStatus.COMPLETED

        except Exception as exc:
            status = ExperimentStatus.FAILED
            error = f"{type(exc).__name__}: {exc}"
            warnings.append(
                "Эксперимент завершился исключением; сохранены доступные результаты."
            )

        result = ExperimentResult(
            experiment_id=experiment_id,
            created_at=created_at,
            status=status,
            config=self.config,
            hardware=hardware_data,
            software=software_data,
            cases=tuple(cases),
            predictions=tuple(predictions),
            measurements=tuple(measurements),
            comparisons=tuple(comparisons),
            decisions=tuple(decisions),
            calibration_summary=tuple(self.performance_model.calibration_summary()),
            warnings=tuple(warnings),
            error=error,
        )

        self.storage.save(result)
        return result

    def _build_cases(self) -> list[ExperimentCase]:
        """Создаёт детерминированный набор calibration/validation cases."""
        cases: list[ExperimentCase] = []

        for split, sizes in (
            (ExperimentSplit.CALIBRATION, self.config.calibration_sizes),
            (ExperimentSplit.VALIDATION, self.config.validation_sizes),
        ):
            for size in sizes:
                seed_material = (
                    f"{self.config.random_seed}:{split.value}:{size}"
                ).encode("utf-8")
                case_seed = int.from_bytes(
                    hashlib.blake2b(
                        seed_material,
                        digest_size=8,
                    ).digest(),
                    byteorder="big",
                )

                for batch_size in self.config.batch_sizes:
                    cases.append(
                        ExperimentCase(
                            split=split,
                            matrix_m=size,
                            matrix_k=size,
                            matrix_n=size,
                            batch_size=batch_size,
                            dtype=self.config.dtype,
                            seed=case_seed,
                        )
                    )

        return cases

    def _create_matrices(
        self,
        case: ExperimentCase,
    ) -> tuple[Any, Any]:
        """Создаёт одинаковые детерминированные CPU float64-матрицы."""
        generator = self.torch.Generator(device="cpu")
        generator.manual_seed(case.seed)

        left = self.torch.randn(
            (case.matrix_m, case.matrix_k),
            dtype=self.torch.float64,
            generator=generator,
            device="cpu",
        )

        right = self.torch.randn(
            (case.matrix_k, case.matrix_n),
            dtype=self.torch.float64,
            generator=generator,
            device="cpu",
        )

        return left, right

    @staticmethod
    def _task_parameters(case: ExperimentCase) -> Any:
        """Создаёт параметры задачи для штатного TaskAnalyzer."""
        from services.compute.scheduler.task_analyzer import TaskParameters

        return TaskParameters(
            matrix_m=case.matrix_m,
            matrix_k=case.matrix_k,
            matrix_n=case.matrix_n,
        )

    def _warmup_cuda(
        self,
        case: ExperimentCase,
        left: Any,
        right: Any,
        *,
        cuda_available: bool,
        warnings: list[str],
    ) -> None:
        """Выполняет CUDA warmup без передачи отчётов в trainer."""
        if not cuda_available or self.config.cuda_warmup == 0:
            return

        for index in range(self.config.cuda_warmup):
            try:
                report = self.coordinator.execute(
                    left,
                    right,
                    requested_backend=ComputeBackend.CUDA,
                    batch_size=case.batch_size,
                )

                if not self._is_calibration_eligible(
                    case,
                    report,
                    ComputeBackend.CUDA,
                ):
                    message = report.execution.error_message or (
                        "CUDA warmup не прошёл строгую проверку."
                    )
                    warnings.append(
                        f"CUDA warmup {case.case_id}, iteration={index}: {message}"
                    )
                    break

            except Exception as exc:
                warnings.append(
                    f"CUDA warmup {case.case_id}, iteration={index}: "
                    f"{type(exc).__name__}: {exc}"
                )
                break

    def _measure(
        self,
        case: ExperimentCase,
        left: Any,
        right: Any,
        backend: ComputeBackend,
        repeat_index: int,
    ) -> tuple[ExperimentMeasurement, MatrixExecutionReport | None]:
        """Выполняет одно измерение и преобразует его в модель эксперимента."""
        try:
            report = self.coordinator.execute(
                left,
                right,
                requested_backend=backend,
                batch_size=case.batch_size,
            )
        except Exception as exc:
            return (
                self._failed_measurement(
                    case,
                    backend,
                    repeat_index,
                    f"{type(exc).__name__}: {exc}",
                ),
                None,
            )

        execution = report.execution

        fallback_used = bool(
            execution.fallback_used
            or (
                report.actual_backend is not None
                and report.actual_backend is not backend
            )
        )

        if fallback_used:
            measurement = ExperimentMeasurement(
                case_id=case.case_id,
                requested_backend=backend,
                actual_backend=report.actual_backend,
                status=MeasurementStatus.FALLBACK,
                repeat_index=repeat_index,
                operations=case.operations,
                input_bytes=sum(item.input_bytes for item in report.batch_executions),
                output_bytes=sum(item.output_bytes for item in report.batch_executions),
                fallback_used=True,
                error=execution.error_message,
            )
            return measurement, report

        if not self._is_calibration_eligible(
            case,
            report,
            backend,
        ):
            return (
                self._failed_measurement(
                    case,
                    backend,
                    repeat_index,
                    execution.error_message
                    or "Отчёт не прошёл строгую проверку измерения.",
                ),
                report,
            )

        input_bytes = sum(item.input_bytes for item in report.batch_executions)
        output_bytes = sum(item.output_bytes for item in report.batch_executions)

        measurement = ExperimentMeasurement(
            case_id=case.case_id,
            requested_backend=backend,
            actual_backend=backend,
            status=MeasurementStatus.SUCCESS,
            repeat_index=repeat_index,
            operations=report.operations,
            input_bytes=input_bytes,
            output_bytes=output_bytes,
            compute_seconds=report.compute_seconds,
            transfer_seconds=report.transfer_seconds,
            total_seconds=report.adapter_total_seconds,
            peak_memory_bytes=report.peak_memory_bytes,
            fallback_used=False,
        )

        return measurement, report

    @staticmethod
    def _is_calibration_eligible(
        case: ExperimentCase,
        report: MatrixExecutionReport,
        expected_backend: ComputeBackend,
    ) -> bool:
        """Проверяет отчёт до передачи в MatrixPerformanceTrainer."""
        if not isinstance(report, MatrixExecutionReport):
            return False

        if not isinstance(case, ExperimentCase):
            return False

        execution = report.execution

        if not report.successful:
            return False

        if execution.result_validated is not True:
            return False

        if execution.fallback_used is not False:
            return False

        if execution.requested_backend is not expected_backend:
            return False

        if execution.actual_backend is not expected_backend:
            return False

        if report.actual_backend is not expected_backend:
            return False

        if not report.batch_executions:
            return False

        if report.analysis.item_count != (
            case.matrix_m * case.matrix_k * case.matrix_n
        ):
            return False

        if report.analysis.estimated_operations != case.operations:
            return False

        if report.operations != case.operations:
            return False

        expected_batch_count = (case.matrix_m + case.batch_size - 1) // case.batch_size

        if len(report.batch_executions) != expected_batch_count:
            return False

        operations_per_row = 2 * case.matrix_k * case.matrix_n

        if operations_per_row <= 0:
            return False

        processed_rows = 0

        for index, batch in enumerate(report.batch_executions):
            if batch.backend is not expected_backend:
                return False

            if not all(
                isfinite(value)
                for value in (
                    batch.compute_seconds,
                    batch.transfer_seconds,
                    batch.total_seconds,
                )
            ):
                return False

            if batch.operations <= 0:
                return False

            if batch.operations % operations_per_row != 0:
                return False

            batch_rows = batch.operations // operations_per_row

            expected_rows = min(
                case.batch_size,
                case.matrix_m - index * case.batch_size,
            )

            if batch_rows != expected_rows:
                return False

            processed_rows += batch_rows

        if processed_rows != case.matrix_m:
            return False

        if not all(
            isfinite(value)
            for value in (
                report.compute_seconds,
                report.transfer_seconds,
                report.adapter_total_seconds,
            )
        ):
            return False

        if report.compute_seconds <= 0:
            return False

        if report.adapter_total_seconds + 1e-12 < (
            report.compute_seconds + report.transfer_seconds
        ):
            return False

        return True

    def _failed_measurement(
        self,
        case: ExperimentCase,
        backend: ComputeBackend,
        repeat_index: int,
        error: str,
    ) -> ExperimentMeasurement:
        """Создаёт FAILED measurement без недостоверных временных метрик."""
        return ExperimentMeasurement(
            case_id=case.case_id,
            requested_backend=backend,
            actual_backend=None,
            status=MeasurementStatus.FAILED,
            repeat_index=repeat_index,
            operations=case.operations,
            input_bytes=0,
            output_bytes=0,
            fallback_used=False,
            error=error,
        )

    def _skipped_measurements(
        self,
        case: ExperimentCase,
        backend: ComputeBackend,
    ) -> list[ExperimentMeasurement]:
        """Создаёт записи SKIPPED для недоступного backend."""
        return [
            ExperimentMeasurement(
                case_id=case.case_id,
                requested_backend=backend,
                actual_backend=None,
                status=MeasurementStatus.SKIPPED,
                repeat_index=repeat_index,
                operations=case.operations,
                input_bytes=0,
                output_bytes=0,
                fallback_used=False,
                error="CUDA недоступна на данном компьютере.",
            )
            for repeat_index in range(self.config.repeats)
        ]

    def _software_metadata(self) -> dict[str, Any]:
        """Собирает версии локального программного окружения."""
        return {
            "python_version": sys.version,
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "torch_version": str(getattr(self.torch, "__version__", "unknown")),
            "experiment_runner": "prediction_accuracy_v1",
        }
