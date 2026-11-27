
"""Тесты обучения PerformanceModel на матричных вычислениях."""

from __future__ import annotations


import pytest

from services.compute.scheduler.execution_types import (
    ExecutionErrorCode,
    ExecutionMetrics,
    ExecutionResult,
    ExecutionStatus,
)
from services.compute.scheduler.matrix_coordinator import (
    MatrixExecutionReport,
)
from services.compute.scheduler.matrix_performance_trainer import (
    MatrixPerformanceTrainer,
)
from services.compute.scheduler.performance_model import (
    PerformanceModel,
)
from services.compute.scheduler.pytorch_adapter import MatmulExecution
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
def analyzer() -> TaskAnalyzer:
    """Создаёт анализатор задач со стандартным реестром."""
    return TaskAnalyzer(create_default_registry())


@pytest.fixture
def model() -> PerformanceModel:
    """Создаёт пустую модель производительности."""
    return PerformanceModel()


@pytest.fixture
def trainer(model: PerformanceModel) -> MatrixPerformanceTrainer:
    """Создаёт тренер с тестовой моделью."""
    return MatrixPerformanceTrainer(model)


@pytest.fixture
def matrix_analysis(analyzer: TaskAnalyzer):
    """Создаёт анализ матричного умножения 4 × 3 на 3 × 2."""
    return analyzer.analyze(
        TaskType.MATRIX_MULTIPLICATION,
        TaskParameters(
            matrix_m=4,
            matrix_k=3,
            matrix_n=2,
        ),
    )


def make_measurement(
    backend: ComputeBackend,
    *,
    operations: int = 24,
    input_bytes: int = 128,
    output_bytes: int = 64,
    compute_seconds: float = 0.01,
    transfer_seconds: float = 0.0,
    total_seconds: float = 0.01,
) -> MatmulExecution:
    """Создаёт синтетическое измерение одного пакета."""
    return MatmulExecution(
        result=object(),
        backend=backend,
        compute_seconds=compute_seconds,
        transfer_seconds=transfer_seconds,
        total_seconds=total_seconds,
        peak_memory_bytes=1024,
        operations=operations,
        input_bytes=input_bytes,
        output_bytes=output_bytes,
    )


def make_success_report(
    analysis,
    backend: ComputeBackend = ComputeBackend.CPU,
    measurements: tuple[MatmulExecution, ...] | None = None,
) -> MatrixExecutionReport:
    """Создаёт корректный успешный отчёт выполнения."""
    if measurements is None:
        measurements = (
            make_measurement(
                backend,
                operations=24,
                input_bytes=128,
                output_bytes=32,
                compute_seconds=0.01,
                transfer_seconds=0.0 if backend is ComputeBackend.CPU else 0.002,
                total_seconds=0.01 if backend is ComputeBackend.CPU else 0.013,
            ),
            make_measurement(
                backend,
                operations=24,
                input_bytes=128,
                output_bytes=32,
                compute_seconds=0.02,
                transfer_seconds=0.0 if backend is ComputeBackend.CPU else 0.003,
                total_seconds=0.02 if backend is ComputeBackend.CPU else 0.024,
            ),
        )

    execution = ExecutionResult(
        task_type=TaskType.MATRIX_MULTIPLICATION,
        status=ExecutionStatus.SUCCESS,
        requested_backend=backend,
        actual_backend=backend,
        result=object(),
        metrics=ExecutionMetrics(
            execution_seconds=0.05,
            preparation_seconds=0.001,
            validation_seconds=0.001,
            batch_count=len(measurements),
            processed_items=4,
            peak_memory_bytes=2048,
        ),
        error_code=ExecutionErrorCode.NONE,
        result_validated=True,
    )

    return MatrixExecutionReport(
        analysis=analysis,
        execution=execution,
        batch_executions=measurements,
    )


def make_failed_report(analysis) -> MatrixExecutionReport:
    """Создаёт отчёт неуспешного выполнения."""
    execution = ExecutionResult(
        task_type=TaskType.MATRIX_MULTIPLICATION,
        status=ExecutionStatus.FAILED,
        requested_backend=ComputeBackend.CUDA,
        actual_backend=None,
        result=None,
        metrics=ExecutionMetrics(
            execution_seconds=0.01,
            preparation_seconds=0.001,
            validation_seconds=0.0,
            batch_count=1,
            processed_items=0,
            peak_memory_bytes=1024,
        ),
        error_code=ExecutionErrorCode.EXECUTION_ERROR,
        error_message="Synthetic execution failure",
        result_validated=False,
    )

    return MatrixExecutionReport(
        analysis=analysis,
        execution=execution,
        batch_executions=(),
    )


def test_trainer_requires_performance_model() -> None:
    """Тренер не принимает объект другого типа."""
    with pytest.raises(TypeError, match="PerformanceModel"):
        MatrixPerformanceTrainer(object())  # type: ignore[arg-type]


def test_record_requires_matrix_execution_report(
    trainer: MatrixPerformanceTrainer,
) -> None:
    """Метод record проверяет тип отчёта."""
    with pytest.raises(TypeError, match="MatrixExecutionReport"):
        trainer.record(object())  # type: ignore[arg-type]


def test_successful_cpu_report_adds_one_sample(
    trainer: MatrixPerformanceTrainer,
    model: PerformanceModel,
    matrix_analysis,
) -> None:
    """Успешное CPU-выполнение добавляет один образец."""
    report = make_success_report(
        matrix_analysis,
        ComputeBackend.CPU,
    )

    result = trainer.record(report)

    assert result is True
    assert model.sample_count == 1


def test_cpu_sample_contains_aggregated_metrics(
    trainer: MatrixPerformanceTrainer,
    model: PerformanceModel,
    matrix_analysis,
) -> None:
    """Метрики нескольких пакетов объединяются в один образец."""
    report = make_success_report(
        matrix_analysis,
        ComputeBackend.CPU,
    )

    assert trainer.record(report) is True

    samples = model.get_samples(
        TaskType.MATRIX_MULTIPLICATION,
        ComputeBackend.CPU,
    )

    assert len(samples) == 1

    sample = samples[0]

    assert sample.operations == 48
    assert sample.input_bytes == 256
    assert sample.output_bytes == 64
    assert sample.compute_seconds == pytest.approx(0.03)
    assert sample.transfer_seconds == pytest.approx(0.0)
    assert sample.total_seconds == pytest.approx(0.03)
    assert sample.backend is ComputeBackend.CPU
    assert sample.task_type is TaskType.MATRIX_MULTIPLICATION


def test_successful_cuda_report_adds_cuda_sample(
    trainer: MatrixPerformanceTrainer,
    model: PerformanceModel,
    matrix_analysis,
) -> None:
    """Успешное CUDA-выполнение обучает CUDA-модель."""
    report = make_success_report(
        matrix_analysis,
        ComputeBackend.CUDA,
    )

    result = trainer.record(report)

    assert result is True
    assert model.sample_count == 1

    samples = model.get_samples(
        TaskType.MATRIX_MULTIPLICATION,
        ComputeBackend.CUDA,
    )

    assert len(samples) == 1

    sample = samples[0]

    assert sample.backend is ComputeBackend.CUDA
    assert sample.operations == 48
    assert sample.compute_seconds == pytest.approx(0.03)
    assert sample.transfer_seconds == pytest.approx(0.005)
    assert sample.total_seconds == pytest.approx(0.037)


def test_cuda_sample_preserves_transfer_measurements(
    trainer: MatrixPerformanceTrainer,
    model: PerformanceModel,
    matrix_analysis,
) -> None:
    """В CUDA-образце учитывается время передачи данных."""
    report = make_success_report(
        matrix_analysis,
        ComputeBackend.CUDA,
    )

    trainer.record(report)

    sample = model.get_samples(
        TaskType.MATRIX_MULTIPLICATION,
        ComputeBackend.CUDA,
    )[0]

    assert sample.transfer_seconds > 0
    assert sample.total_seconds >= (
        sample.compute_seconds + sample.transfer_seconds
    )


def test_cpu_and_cuda_samples_are_stored_separately(
    trainer: MatrixPerformanceTrainer,
    model: PerformanceModel,
    matrix_analysis,
) -> None:
    """CPU и CUDA измерения не смешиваются."""
    cpu_report = make_success_report(
        matrix_analysis,
        ComputeBackend.CPU,
    )
    cuda_report = make_success_report(
        matrix_analysis,
        ComputeBackend.CUDA,
    )

    assert trainer.record(cpu_report) is True
    assert trainer.record(cuda_report) is True

    assert model.sample_count == 2

    cpu_samples = model.get_samples(
        TaskType.MATRIX_MULTIPLICATION,
        ComputeBackend.CPU,
    )
    cuda_samples = model.get_samples(
        TaskType.MATRIX_MULTIPLICATION,
        ComputeBackend.CUDA,
    )

    assert len(cpu_samples) == 1
    assert len(cuda_samples) == 1

    assert cpu_samples[0].backend is ComputeBackend.CPU
    assert cuda_samples[0].backend is ComputeBackend.CUDA


def test_failed_report_does_not_train_model(
    trainer: MatrixPerformanceTrainer,
    model: PerformanceModel,
    matrix_analysis,
) -> None:
    """Неуспешное выполнение не добавляет образец."""
    report = make_failed_report(matrix_analysis)

    result = trainer.record(report)

    assert result is False
    assert model.sample_count == 0


def test_failed_report_does_not_require_batch_measurements(
    trainer: MatrixPerformanceTrainer,
    matrix_analysis,
) -> None:
    """Для неуспешного отчёта пустой список пакетов допустим."""
    report = make_failed_report(matrix_analysis)

    assert report.batch_executions == ()
    assert trainer.record(report) is False



def test_successful_report_requires_validated_result(
    trainer: MatrixPerformanceTrainer,
    matrix_analysis,
) -> None:
    """Непроверенный результат не может обучать модель."""
    report = make_success_report(
        matrix_analysis,
        ComputeBackend.CPU,
    )

    # Намеренно нарушаем инвариант ExecutionResult
    # только для проверки защитной ветви тренера.
    object.__setattr__(
        report.execution,
        "result_validated",
        False,
    )

    with pytest.raises(ValueError, match="непроверенном"):
        trainer.record(report)




def test_successful_report_requires_backend(
    trainer: MatrixPerformanceTrainer,
    matrix_analysis,
) -> None:
    """Успешный отчёт должен содержать фактический backend."""
    report = make_success_report(
        matrix_analysis,
        ComputeBackend.CPU,
    )

    # Намеренно нарушаем инвариант ExecutionResult
    # только для проверки защитной ветви тренера.
    object.__setattr__(
        report.execution,
        "actual_backend",
        None,
    )

    with pytest.raises(ValueError, match="backend"):
        trainer.record(report)



def test_successful_report_requires_batch_measurements(
    trainer: MatrixPerformanceTrainer,
    matrix_analysis,
) -> None:
    """Успешный отчёт не может содержать пустые измерения."""
    report = make_success_report(
        matrix_analysis,
        ComputeBackend.CPU,
        measurements=(),
    )

    with pytest.raises(ValueError, match="измерений"):
        trainer.record(report)


def test_measurement_must_be_matmul_execution(
    trainer: MatrixPerformanceTrainer,
    matrix_analysis,
) -> None:
    """Каждое измерение должно иметь ожидаемый тип."""
    report = make_success_report(
        matrix_analysis,
        ComputeBackend.CPU,
        measurements=(object(),),  # type: ignore[arg-type]
    )

    with pytest.raises(TypeError, match="MatmulExecution"):
        trainer.record(report)


def test_measurement_backend_must_match_execution_backend(
    trainer: MatrixPerformanceTrainer,
    model: PerformanceModel,
    matrix_analysis,
) -> None:
    """Нельзя смешивать CPU и CUDA пакеты в одном отчёте."""
    measurements = (
        make_measurement(
            ComputeBackend.CPU,
            operations=24,
        ),
        make_measurement(
            ComputeBackend.CUDA,
            operations=24,
        ),
    )

    report = make_success_report(
        matrix_analysis,
        ComputeBackend.CPU,
        measurements=measurements,
    )

    with pytest.raises(ValueError, match="Backend измерения"):
        trainer.record(report)

    assert model.sample_count == 0


def test_operation_count_must_match_analysis(
    trainer: MatrixPerformanceTrainer,
    model: PerformanceModel,
    matrix_analysis,
) -> None:
    """Количество измеренных операций сверяется с анализом."""
    measurements = (
        make_measurement(
            ComputeBackend.CPU,
            operations=20,
        ),
        make_measurement(
            ComputeBackend.CPU,
            operations=20,
        ),
    )

    report = make_success_report(
        matrix_analysis,
        ComputeBackend.CPU,
        measurements=measurements,
    )

    with pytest.raises(ValueError, match="операций"):
        trainer.record(report)

    assert model.sample_count == 0


def test_invalid_report_does_not_partially_modify_model(
    trainer: MatrixPerformanceTrainer,
    model: PerformanceModel,
    matrix_analysis,
) -> None:
    """Ошибка проверки не должна частично менять модель."""
    valid_report = make_success_report(
        matrix_analysis,
        ComputeBackend.CPU,
    )

    invalid_measurements = (
        make_measurement(
            ComputeBackend.CPU,
            operations=24,
        ),
        make_measurement(
            ComputeBackend.CUDA,
            operations=24,
        ),
    )

    invalid_report = make_success_report(
        matrix_analysis,
        ComputeBackend.CPU,
        measurements=invalid_measurements,
    )

    assert trainer.record(valid_report) is True
    assert model.sample_count == 1

    with pytest.raises(ValueError):
        trainer.record(invalid_report)

    assert model.sample_count == 1


def test_multiple_reports_increase_sample_count(
    trainer: MatrixPerformanceTrainer,
    model: PerformanceModel,
    matrix_analysis,
) -> None:
    """Каждый успешный отчёт добавляет ровно один образец."""
    reports = [
        make_success_report(matrix_analysis, ComputeBackend.CPU),
        make_success_report(matrix_analysis, ComputeBackend.CPU),
        make_success_report(matrix_analysis, ComputeBackend.CPU),
    ]

    for report in reports:
        assert trainer.record(report) is True

    assert model.sample_count == 3

    samples = model.get_samples(
        TaskType.MATRIX_MULTIPLICATION,
        ComputeBackend.CPU,
    )

    assert len(samples) == 3


def test_cuda_fallback_report_trains_only_cpu_backend(
    trainer: MatrixPerformanceTrainer,
    model: PerformanceModel,
    matrix_analysis,
) -> None:
    """
    При CUDA fallback обучается только фактически успешный CPU backend.

    В отчёт передаются только измерения успешного CPU-выполнения.
    """
    cpu_measurements = (
        make_measurement(
            ComputeBackend.CPU,
            operations=24,
            compute_seconds=0.01,
            total_seconds=0.01,
        ),
        make_measurement(
            ComputeBackend.CPU,
            operations=24,
            compute_seconds=0.02,
            total_seconds=0.02,
        ),
    )

    execution = ExecutionResult(
        task_type=TaskType.MATRIX_MULTIPLICATION,
        status=ExecutionStatus.SUCCESS,
        requested_backend=ComputeBackend.CUDA,
        actual_backend=ComputeBackend.CPU,
        result=object(),
        metrics=ExecutionMetrics(
            execution_seconds=0.04,
            preparation_seconds=0.001,
            validation_seconds=0.001,
            batch_count=2,
            processed_items=4,
            peak_memory_bytes=2048,
        ),
        error_code=ExecutionErrorCode.NONE,
        fallback_used=True,
        result_validated=True,
    )

    report = MatrixExecutionReport(
        analysis=matrix_analysis,
        execution=execution,
        batch_executions=cpu_measurements,
    )

    assert report.execution.fallback_used is True
    assert report.actual_backend is ComputeBackend.CPU

    assert trainer.record(report) is True

    assert model.sample_count == 1

    cpu_samples = model.get_samples(
        TaskType.MATRIX_MULTIPLICATION,
        ComputeBackend.CPU,
    )
    cuda_samples = model.get_samples(
        TaskType.MATRIX_MULTIPLICATION,
        ComputeBackend.CUDA,
    )

    assert len(cpu_samples) == 1
    assert len(cuda_samples) == 0

    assert cpu_samples[0].backend is ComputeBackend.CPU
    assert cpu_samples[0].transfer_seconds == pytest.approx(0.0)


def test_record_returns_false_for_cancelled_execution(
    trainer: MatrixPerformanceTrainer,
    model: PerformanceModel,
    matrix_analysis,
) -> None:
    """Отменённое выполнение не обучает модель."""
    execution = ExecutionResult(
        task_type=TaskType.MATRIX_MULTIPLICATION,
        status=ExecutionStatus.CANCELLED,
        requested_backend=ComputeBackend.CPU,
        actual_backend=None,
        result=None,
        metrics=ExecutionMetrics(
            execution_seconds=0.01,
            preparation_seconds=0.001,
            validation_seconds=0.0,
            batch_count=1,
            processed_items=0,
            peak_memory_bytes=1024,
        ),
        error_code=ExecutionErrorCode.CANCELLED,
        error_message="Execution cancelled",
        result_validated=False,
    )

    report = MatrixExecutionReport(
        analysis=matrix_analysis,
        execution=execution,
        batch_executions=(),
    )

    assert trainer.record(report) is False
    assert model.sample_count == 0


def test_record_does_not_mix_different_task_types(
    trainer: MatrixPerformanceTrainer,
    model: PerformanceModel,
    matrix_analysis,
) -> None:
    """Образец матричного умножения сохраняется только для своего типа."""
    report = make_success_report(
        matrix_analysis,
        ComputeBackend.CPU,
    )

    assert trainer.record(report) is True

    assert len(
        model.get_samples(
            TaskType.MATRIX_MULTIPLICATION,
            ComputeBackend.CPU,
        )
    ) == 1

    assert len(
        model.get_samples(
            TaskType.STATISTICAL_ANALYSIS,
            ComputeBackend.CPU,
        )
    ) == 0
