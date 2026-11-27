
"""Реестр вычислительных задач LAIE."""

from __future__ import annotations

from services.compute.scheduler.task_types import (
    ComputeBackend,
    PrecisionRequirement,
    TaskDefinition,
    TaskType,
    WorkloadKind,
)


class TaskAlreadyRegisteredError(ValueError):
    """Задача с таким типом уже зарегистрирована."""


class TaskNotFoundError(LookupError):
    """Запрашиваемый тип задачи отсутствует в реестре."""


class TaskRegistry:
    """Централизованный каталог доступных вычислительных задач."""

    def __init__(self) -> None:
        self._tasks: dict[TaskType, TaskDefinition] = {}

    def register(self, definition: TaskDefinition) -> None:
        """Регистрирует описание новой вычислительной задачи."""
        if not isinstance(definition, TaskDefinition):
            raise TypeError("definition должен быть TaskDefinition")

        if definition.task_type in self._tasks:
            raise TaskAlreadyRegisteredError(
                f"Задача уже зарегистрирована: "
                f"{definition.task_type.value}"
            )

        self._tasks[definition.task_type] = definition

    def get(self, task_type: TaskType) -> TaskDefinition:
        """Возвращает описание задачи по её типу."""
        if not isinstance(task_type, TaskType):
            raise TypeError("task_type должен быть элементом TaskType")

        try:
            return self._tasks[task_type]
        except KeyError as exc:
            raise TaskNotFoundError(
                f"Неизвестный тип задачи: {task_type.value}"
            ) from exc

    def contains(self, task_type: TaskType) -> bool:
        """Проверяет наличие типа задачи в реестре."""
        if not isinstance(task_type, TaskType):
            raise TypeError("task_type должен быть элементом TaskType")

        return task_type in self._tasks

    def list_all(self) -> tuple[TaskDefinition, ...]:
        """Возвращает неизменяемый снимок зарегистрированных задач."""
        return tuple(self._tasks.values())

    def list_types(self) -> tuple[TaskType, ...]:
        """Возвращает зарегистрированные типы задач."""
        return tuple(self._tasks.keys())

    def __len__(self) -> int:
        return len(self._tasks)

    def __contains__(self, task_type: object) -> bool:
        return (
            isinstance(task_type, TaskType)
            and task_type in self._tasks
        )


def create_default_registry() -> TaskRegistry:
    """Создаёт реестр со стандартными категориями задач LAIE."""
    registry = TaskRegistry()

    definitions = (
        TaskDefinition(
            task_type=TaskType.MATRIX_MULTIPLICATION,
            name="Матричные операции",
            description="Умножение матриц и связанные операции.",
            workload=WorkloadKind.GPU_PARALLEL,
            supported_backends=(
                ComputeBackend.CPU,
                ComputeBackend.CUDA,
            ),
            precision=PrecisionRequirement.FLOAT64,
            parallelizable=True,
            supports_batching=True,
            estimator_key="matrix_multiplication",
        ),
        TaskDefinition(
            task_type=TaskType.STATISTICAL_ANALYSIS,
            name="Статистический анализ",
            description="Статистические расчёты и анализ выборок.",
            workload=WorkloadKind.MIXED,
            supported_backends=(
                ComputeBackend.CPU,
                ComputeBackend.CUDA,
            ),
            precision=PrecisionRequirement.FLOAT64,
            parallelizable=True,
            supports_batching=True,
            estimator_key="statistical_analysis",
        ),
        TaskDefinition(
            task_type=TaskType.COMBINATION_GENERATION,
            name="Генерация комбинаций",
            description="Формирование и фильтрация комбинаций.",
            workload=WorkloadKind.CPU_BOUND,
            supported_backends=(ComputeBackend.CPU,),
            precision=PrecisionRequirement.EXACT,
            parallelizable=True,
            supports_batching=True,
            estimator_key="combination_generation",
        ),
        TaskDefinition(
            task_type=TaskType.MONTE_CARLO,
            name="Моделирование Монте-Карло",
            description="Массовые независимые случайные симуляции.",
            workload=WorkloadKind.GPU_PARALLEL,
            supported_backends=(
                ComputeBackend.CPU,
                ComputeBackend.CUDA,
            ),
            precision=PrecisionRequirement.FLOAT64,
            parallelizable=True,
            supports_batching=True,
            estimator_key="monte_carlo",
        ),
        TaskDefinition(
            task_type=TaskType.GENETIC_OPTIMIZATION,
            name="Генетическая оптимизация",
            description="Эволюционный поиск и оптимизация решений.",
            workload=WorkloadKind.MIXED,
            supported_backends=(
                ComputeBackend.CPU,
                ComputeBackend.CUDA,
            ),
            precision=PrecisionRequirement.FLOAT64,
            parallelizable=True,
            supports_batching=True,
            estimator_key="genetic_optimization",
        ),
        TaskDefinition(
            task_type=TaskType.MODEL_TRAINING,
            name="Обучение моделей",
            description="Обучение локальных статистических и ML-моделей.",
            workload=WorkloadKind.GPU_PARALLEL,
            supported_backends=(
                ComputeBackend.CPU,
                ComputeBackend.CUDA,
            ),
            precision=PrecisionRequirement.FLOAT32,
            parallelizable=True,
            supports_batching=True,
            estimator_key="model_training",
        ),
        TaskDefinition(
            task_type=TaskType.HYPOTHESIS_TESTING,
            name="Проверка гипотез",
            description="Статистическая проверка исследовательских гипотез.",
            workload=WorkloadKind.CPU_BOUND,
            supported_backends=(ComputeBackend.CPU,),
            precision=PrecisionRequirement.FLOAT64,
            parallelizable=True,
            supports_batching=True,
            estimator_key="hypothesis_testing",
        ),
    )

    for definition in definitions:
        registry.register(definition)

    return registry
