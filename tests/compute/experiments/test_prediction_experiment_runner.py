"""Проверки строгого допуска отчётов к калибровке эксперимента."""

from dataclasses import replace

import pytest

from services.compute.experiments.experiment_models import (
    ExperimentCase,
    ExperimentSplit,
)
from services.compute.experiments.prediction_experiment_runner import (
    PredictionExperimentRunner,
)
from services.compute.scheduler.execution_types import (
    ExecutionMetrics,
    ExecutionResult,
    ExecutionStatus,
)
from services.compute.scheduler.matrix_coordinator import (
    MatrixExecutionReport,
)
from services.compute.scheduler.pytorch_adapter import MatmulExecution
from services.compute.scheduler.task_analyzer import (
    TaskAnalyzer,
    TaskParameters,
)
from services.compute.scheduler.task_registry import create_default_registry
from services.compute.scheduler.performance_model import (
    PerformanceModel,
    PerformanceSample,
)
from services.compute.scheduler.task_types import (
    ComputeBackend,
    TaskType,
)


@pytest.fixture
def calibration_report():
    """Создаёт настоящий успешный отчёт без выполнения матриц."""
    case = ExperimentCase(
        split=ExperimentSplit.CALIBRATION,
        matrix_m=5,
        matrix_k=3,
        matrix_n=4,
        batch_size=2,
        dtype="float64",
        seed=42,
    )

    analyzer = TaskAnalyzer(create_default_registry())
    analysis = analyzer.analyze(
        TaskType.MATRIX_MULTIPLICATION,
        TaskParameters(
            matrix_m=case.matrix_m,
            matrix_k=case.matrix_k,
            matrix_n=case.matrix_n,
        ),
    )

    operations_per_row = 2 * case.matrix_k * case.matrix_n

    batches = tuple(
        MatmulExecution(
            result=None,
            backend=ComputeBackend.CPU,
            compute_seconds=0.01,
            transfer_seconds=0.0,
            total_seconds=0.01,
            peak_memory_bytes=None,
            operations=rows * operations_per_row,
            input_bytes=1,
            output_bytes=1,
        )
        for rows in (2, 2, 1)
    )

    execution = ExecutionResult(
        task_type=TaskType.MATRIX_MULTIPLICATION,
        status=ExecutionStatus.SUCCESS,
        requested_backend=ComputeBackend.CPU,
        actual_backend=ComputeBackend.CPU,
        result=None,
        metrics=ExecutionMetrics(
            execution_seconds=0.03,
            preparation_seconds=0.0,
            validation_seconds=0.0,
            batch_count=3,
            processed_items=case.matrix_m,
        ),
        result_validated=True,
    )

    return case, MatrixExecutionReport(
        analysis=analysis,
        execution=execution,
        batch_executions=batches,
    )


def test_calibration_gate_accepts_valid_report(calibration_report):
    """Корректный отчёт CPU допускается к калибровке."""
    case, report = calibration_report

    assert PredictionExperimentRunner._is_calibration_eligible(
        case,
        report,
        ComputeBackend.CPU,
    )


@pytest.mark.parametrize(
    "metric_changes",
    (
        {"compute_seconds": float("nan")},
        {"compute_seconds": float("inf"), "total_seconds": float("inf")},
        {"total_seconds": float("nan")},
        {"total_seconds": float("inf")},
    ),
)
def test_calibration_gate_rejects_non_finite_batch_metrics(
    calibration_report,
    metric_changes,
):
    """Gate ????????? NaN ? Infinity ? ???????? ???????."""
    case, report = calibration_report
    invalid_batches = (
        replace(report.batch_executions[0], **metric_changes),
        *report.batch_executions[1:],
    )
    invalid_report = replace(
        report,
        batch_executions=invalid_batches,
    )

    assert not PredictionExperimentRunner._is_calibration_eligible(
        case,
        invalid_report,
        ComputeBackend.CPU,
    )


def test_calibration_gate_rejects_overflowed_aggregate_metrics(
    calibration_report,
):
    """Gate ????????? ???????????? ?????????????? ????????? ??????."""
    case, report = calibration_report
    invalid_batches = tuple(
        replace(
            batch,
            compute_seconds=1e308,
            total_seconds=1e308,
        )
        for batch in report.batch_executions
    )
    invalid_report = replace(
        report,
        batch_executions=invalid_batches,
    )

    assert not PredictionExperimentRunner._is_calibration_eligible(
        case,
        invalid_report,
        ComputeBackend.CPU,
    )


@pytest.mark.parametrize(
    "execution_changes",
    (
        {"fallback_used": True},
        {"requested_backend": ComputeBackend.CUDA},
        {"actual_backend": ComputeBackend.CUDA},
    ),
)
def test_calibration_gate_rejects_invalid_execution(
    calibration_report,
    execution_changes,
):
    """Gate отклоняет fallback и нарушения контракта выполнения."""
    case, report = calibration_report
    invalid_execution = replace(
        report.execution,
        **execution_changes,
    )
    invalid_report = replace(
        report,
        execution=invalid_execution,
    )

    assert not PredictionExperimentRunner._is_calibration_eligible(
        case,
        invalid_report,
        ComputeBackend.CPU,
    )


def test_execution_result_rejects_unvalidated_success(
    calibration_report,
):
    """Проверяет инвариант успешного выполнения."""
    _, report = calibration_report

    with pytest.raises(ValueError):
        replace(
            report.execution,
            result_validated=False,
        )


@pytest.mark.parametrize(
    "analysis_changes",
    (
        {"item_count": 4},
        {"estimated_operations": 1},
    ),
)
def test_calibration_gate_rejects_incorrect_analysis(
    calibration_report,
    analysis_changes,
):
    """Gate сверяет анализ задачи с исходной конфигурацией."""
    case, report = calibration_report
    invalid_report = replace(
        report,
        analysis=replace(
            report.analysis,
            **analysis_changes,
        ),
    )

    assert not PredictionExperimentRunner._is_calibration_eligible(
        case,
        invalid_report,
        ComputeBackend.CPU,
    )


def test_calibration_gate_rejects_missing_batch(calibration_report):
    """Gate не принимает отчёт с неполным количеством пакетов."""
    case, report = calibration_report
    invalid_report = replace(
        report,
        batch_executions=report.batch_executions[:-1],
    )

    assert not PredictionExperimentRunner._is_calibration_eligible(
        case,
        invalid_report,
        ComputeBackend.CPU,
    )


def test_calibration_gate_rejects_incorrect_batch_operations(
    calibration_report,
):
    """Gate проверяет точное количество строк в каждом пакете."""
    case, report = calibration_report
    invalid_batches = (
        replace(report.batch_executions[0], operations=24),
        *report.batch_executions[1:],
    )
    invalid_report = replace(
        report,
        batch_executions=invalid_batches,
    )

    assert not PredictionExperimentRunner._is_calibration_eligible(
        case,
        invalid_report,
        ComputeBackend.CPU,
    )


def test_calibration_gate_rejects_wrong_expected_backend(
    calibration_report,
):
    """Gate требует совпадения backend отчёта и ожидаемого backend."""
    case, report = calibration_report

    assert not PredictionExperimentRunner._is_calibration_eligible(
        case,
        report,
        ComputeBackend.CUDA,
    )


def test_calibration_gate_rejects_non_report_and_non_case(
    calibration_report,
):
    """Gate отклоняет объекты, не соответствующие типизированному контракту."""
    case, report = calibration_report

    assert not PredictionExperimentRunner._is_calibration_eligible(
        case,
        object(),
        ComputeBackend.CPU,
    )

    assert not PredictionExperimentRunner._is_calibration_eligible(
        object(),
        report,
        ComputeBackend.CPU,
    )


# RUNNER_LIFECYCLE_TEST
def test_runner_keeps_validation_out_of_training_and_saves_report(
    tmp_path,
    monkeypatch,
):
    """Проверяет изоляцию validation и сохранение полного JSON-отчёта."""
    import importlib

    from services.compute.experiments.experiment_models import (
        ExperimentConfig,
        ExperimentStatus,
        MeasurementStatus,
    )
    from services.compute.experiments.experiment_storage import (
        ExperimentStorage,
    )
    from services.compute.scheduler.execution_types import (
        ExecutionMetrics,
        ExecutionResult,
        ExecutionStatus,
    )
    from services.compute.scheduler.hardware_monitor import (
        HardwareMonitor,
        HardwareSnapshot,
    )
    from services.compute.scheduler.matrix_coordinator import (
        MatrixExecutionReport,
    )
    from services.compute.scheduler.pytorch_adapter import MatmulExecution
    from services.compute.scheduler.task_analyzer import TaskParameters

    torch = importlib.import_module("torch")

    config = ExperimentConfig(
        calibration_sizes=(2,),
        validation_sizes=(3,),
        batch_sizes=(1,),
        repeats=2,
        cuda_warmup=0,
        high_confidence_samples=2,
    )

    external_model = PerformanceModel(high_confidence_samples=2)
    external_sample = PerformanceSample(
        task_type=TaskType.MATRIX_MULTIPLICATION,
        backend=ComputeBackend.CPU,
        operations=100,
        input_bytes=64,
        output_bytes=32,
        compute_seconds=0.5,
        transfer_seconds=0.0,
        total_seconds=0.5,
    )
    external_model.add_sample(external_sample)

    storage = ExperimentStorage(tmp_path)
    monitor = HardwareMonitor(torch)

    snapshot = HardwareSnapshot(
        cpu_name="Test CPU",
        cpu_logical_cores=4,
        ram_total_bytes=8 * 1024**3,
        ram_available_bytes=7 * 1024**3,
        cuda_available=False,
        cuda_reason="CUDA disabled in test",
    )

    monkeypatch.setattr(
        monitor,
        "inspect",
        lambda: snapshot,
    )

    runner = PredictionExperimentRunner(
        config,
        torch_module=torch,
        storage=storage,
        hardware_monitor=monitor,
        performance_model=external_model,
    )

    assert runner.performance_model is not external_model
    assert runner.performance_model.sample_count == 0

    # Вместо матриц передаём сам case как тестовый маркер.
    monkeypatch.setattr(
        runner,
        "_create_matrices",
        lambda case: (case, case),
    )

    def fake_execute(
        left,
        right,
        *,
        requested_backend,
        batch_size,
    ):
        case = left

        assert right is case

        analysis = runner.task_analyzer.analyze(
            TaskType.MATRIX_MULTIPLICATION,
            TaskParameters(
                matrix_m=case.matrix_m,
                matrix_k=case.matrix_k,
                matrix_n=case.matrix_n,
            ),
        )

        batches = []

        for start in range(0, case.matrix_m, batch_size):
            rows = min(
                batch_size,
                case.matrix_m - start,
            )

            batches.append(
                MatmulExecution(
                    result=None,
                    backend=requested_backend,
                    compute_seconds=0.01,
                    transfer_seconds=0.0,
                    total_seconds=0.01,
                    peak_memory_bytes=None,
                    operations=(rows * 2 * case.matrix_k * case.matrix_n),
                    input_bytes=1,
                    output_bytes=1,
                )
            )

        execution = ExecutionResult(
            task_type=TaskType.MATRIX_MULTIPLICATION,
            status=ExecutionStatus.SUCCESS,
            requested_backend=requested_backend,
            actual_backend=requested_backend,
            result=None,
            metrics=ExecutionMetrics(
                execution_seconds=len(batches) * 0.01,
                preparation_seconds=0.0,
                validation_seconds=0.0,
                batch_count=len(batches),
                processed_items=case.matrix_m,
            ),
            result_validated=True,
        )

        return MatrixExecutionReport(
            analysis=analysis,
            execution=execution,
            batch_executions=tuple(batches),
        )

    monkeypatch.setattr(
        runner.coordinator,
        "execute",
        fake_execute,
    )

    trained_item_counts = []
    original_record = runner.trainer.record

    def tracked_record(report):
        trained_item_counts.append(report.analysis.item_count)
        return original_record(report)

    monkeypatch.setattr(
        runner.trainer,
        "record",
        tracked_record,
    )

    result = runner.run()

    # Внешняя модель и её исходные данные не затронуты.
    assert external_model.sample_count == 1
    assert external_model.get_samples(
        TaskType.MATRIX_MULTIPLICATION,
        ComputeBackend.CPU,
    ) == (external_sample,)

    # Два успешных CPU-повтора калибровки записаны только в модель Runner.
    assert len(trained_item_counts) == 2
    assert trained_item_counts == [8, 8]
    assert runner.performance_model.sample_count == 2
    assert (
        len(
            runner.performance_model.get_samples(
                TaskType.MATRIX_MULTIPLICATION,
                ComputeBackend.CPU,
            )
        )
        == 2
    )

    # Валидационные отчёты имеют item_count=27 и не обучают модель.
    assert 27 not in trained_item_counts

    assert result.status is ExperimentStatus.PARTIAL
    assert result.error is None

    assert (
        sum(item.status is MeasurementStatus.SUCCESS for item in result.measurements)
        == 4
    )

    assert (
        sum(item.status is MeasurementStatus.SKIPPED for item in result.measurements)
        == 4
    )

    assert len(result.predictions) == 1
    assert result.predictions[0]["case_id"].startswith("validation:")

    # Проверяем реальный JSON-файл через публичный API хранилища.
    report_path = tmp_path / f"{result.experiment_id}.json"
    assert report_path.is_file()

    payload = storage.load(report_path)

    assert payload["experiment_type"] == "prediction_accuracy"
    assert payload["status"] == "partial"
    assert len(payload["measurements"]) == 8
    assert payload["config"]["calibration_sizes"] == [2]
    assert payload["config"]["validation_sizes"] == [3]

    # Повторный запуск не должен повторно обучать модель или сохранять отчёт.
    with pytest.raises(RuntimeError, match="только один раз"):
        runner.run()

    assert runner.performance_model.sample_count == 2


def test_runner_build_cases_uses_reproducible_independent_seeds():
    """Seed зависит от split и размера, но не от batch_size."""
    import importlib

    from services.compute.experiments.experiment_models import ExperimentConfig

    torch = importlib.import_module("torch")
    config = ExperimentConfig(
        calibration_sizes=(2, 4),
        validation_sizes=(3, 5),
        batch_sizes=(1, 2),
        repeats=2,
        cuda_warmup=0,
        random_seed=42,
        high_confidence_samples=2,
    )

    first_runner = PredictionExperimentRunner(
        config,
        torch_module=torch,
    )
    second_runner = PredictionExperimentRunner(
        config,
        torch_module=torch,
    )

    first_cases = first_runner._build_cases()
    second_cases = second_runner._build_cases()

    assert first_cases == second_cases

    seeds_by_group = {}
    for case in first_cases:
        group = (case.split, case.matrix_m)
        seeds_by_group.setdefault(group, set()).add(case.seed)

    # Для одного split и размера seed одинаков при разных batch_size.
    assert all(len(seeds) == 1 for seeds in seeds_by_group.values())

    # Все четыре группы получают независимые seed.
    unique_seeds = {next(iter(seeds)) for seeds in seeds_by_group.values()}
    assert len(unique_seeds) == 4
