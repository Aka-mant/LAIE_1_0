
"""Обучение модели производительности на результатах матричных вычислений."""

from __future__ import annotations

from .execution_types import ExecutionStatus
from .matrix_coordinator import MatrixExecutionReport
from .performance_model import PerformanceModel, PerformanceSample
from .pytorch_adapter import MatmulExecution
from .task_types import ComputeBackend, TaskType


class MatrixPerformanceTrainer:
    """Передаёт проверенные измерения матричных вычислений в PerformanceModel."""

    def __init__(self, model: PerformanceModel) -> None:
        """Инициализирует тренер модели производительности."""
        if not isinstance(model, PerformanceModel):
            raise TypeError("model должен быть экземпляром PerformanceModel")

        self.model = model

    def record(self, report: MatrixExecutionReport) -> bool:
        """
        Записывает одно агрегированное измерение в PerformanceModel.

        Возвращает:
            True, если измерение добавлено.
            False, если вычисление завершилось неуспешно или было отменено.

        Исключения:
            TypeError: если передан объект неправильного типа.
            ValueError: если успешный отчёт содержит некорректные данные.
        """
        if not isinstance(report, MatrixExecutionReport):
            raise TypeError(
                "report должен быть экземпляром MatrixExecutionReport"
            )

        execution = report.execution

        # Неуспешные и отменённые вычисления не должны обучать модель.
        if execution.status is not ExecutionStatus.SUCCESS:
            return False

        # Дополнительная защита от использования непроверенного результата.
        if not execution.result_validated:
            raise ValueError(
                "Нельзя обучать модель на непроверенном результате"
            )

        backend = report.actual_backend

        if not isinstance(backend, ComputeBackend):
            raise ValueError(
                "Успешный отчёт должен содержать фактический backend"
            )

        if execution.actual_backend is not backend:
            raise ValueError(
                "Фактический backend отчёта не совпадает "
                "с backend результата выполнения"
            )

        task_type = report.analysis.task_type

        if task_type is not TaskType.MATRIX_MULTIPLICATION:
            raise ValueError(
                "MatrixPerformanceTrainer поддерживает только "
                "матричное умножение"
            )

        measurements = report.batch_executions

        if not measurements:
            raise ValueError(
                "Успешный отчёт не содержит измерений пакетов"
            )

        # Проверяем каждое измерение до изменения PerformanceModel.
        for index, measurement in enumerate(measurements):
            if not isinstance(measurement, MatmulExecution):
                raise TypeError(
                    f"Измерение пакета {index} должно быть "
                    "экземпляром MatmulExecution"
                )

            if measurement.backend is not backend:
                raise ValueError(
                    f"Backend измерения пакета {index} "
                    "не совпадает с фактическим backend выполнения"
                )

        # Агрегируем показатели всех успешно выполненных пакетов.
        operations = sum(
            measurement.operations for measurement in measurements
        )
        input_bytes = sum(
            measurement.input_bytes for measurement in measurements
        )
        output_bytes = sum(
            measurement.output_bytes for measurement in measurements
        )
        compute_seconds = sum(
            measurement.compute_seconds for measurement in measurements
        )
        transfer_seconds = sum(
            measurement.transfer_seconds for measurement in measurements
        )
        total_seconds = sum(
            measurement.total_seconds for measurement in measurements
        )

        expected_operations = report.analysis.estimated_operations

        if operations != expected_operations:
            raise ValueError(
                "Количество фактически измеренных операций "
                f"({operations}) не совпадает с ожидаемым "
                f"({expected_operations})"
            )

        # PerformanceSample дополнительно проверяет корректность метрик.
        sample = PerformanceSample(
            task_type=task_type,
            backend=backend,
            operations=operations,
            input_bytes=input_bytes,
            output_bytes=output_bytes,
            compute_seconds=compute_seconds,
            transfer_seconds=transfer_seconds,
            total_seconds=total_seconds,
        )

        # Изменение модели происходит только после всех проверок.
        self.model.add_sample(sample)

        return True
