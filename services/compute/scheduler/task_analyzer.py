
"""Анализ сложности и оценка ресурсов вычислительных задач LAIE."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any
import math

from services.compute.scheduler.task_registry import TaskRegistry
from services.compute.scheduler.task_types import (
    ComputeBackend,
    PrecisionRequirement,
    TaskType,
    WorkloadKind,
)


@dataclass(frozen=True, slots=True)
class TaskParameters:
    """Входные параметры для оценки вычислительной задачи.

    Все размеры задаются целыми положительными числами.
    Неиспользуемые для конкретной задачи поля остаются None.
    """

    # Общие параметры
    item_count: int | None = None
    batch_size: int | None = None

    # Матричные операции: M x K умножить на K x N
    matrix_m: int | None = None
    matrix_k: int | None = None
    matrix_n: int | None = None

    # Статистический анализ
    sample_count: int | None = None
    feature_count: int | None = None

    # Генерация комбинаций
    candidate_count: int | None = None
    item_width: int | None = None

    # Монте-Карло
    iterations: int | None = None
    operations_per_iteration: int | None = None
    state_size: int | None = None

    # Генетическая оптимизация
    population_size: int | None = None
    generations: int | None = None
    genome_size: int | None = None

    # Обучение моделей
    epochs: int | None = None

    # Проверка гипотез
    hypothesis_count: int | None = None

    def __post_init__(self) -> None:
        """Проверяет типы и диапазоны входных параметров."""
        for field in fields(self):
            value = getattr(self, field.name)

            if value is None:
                continue

            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"{field.name} должен быть целым числом или None"
                )

            if value <= 0:
                raise ValueError(
                    f"{field.name} должен быть положительным числом"
                )


@dataclass(frozen=True, slots=True)
class TaskAnalysis:
    """Результат анализа вычислительной задачи."""

    task_type: TaskType
    item_count: int
    estimated_operations: int
    estimated_input_bytes: int
    estimated_output_bytes: int
    estimated_workspace_bytes: int
    estimated_memory_bytes: int
    workload: WorkloadKind
    supported_backends: tuple[ComputeBackend, ...]
    precision: PrecisionRequirement
    parallelizable: bool
    supports_batching: bool
    complexity_model: str
    assumptions: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Возвращает сериализуемое представление результата."""
        return {
            "task_type": self.task_type.value,
            "item_count": self.item_count,
            "estimated_operations": self.estimated_operations,
            "estimated_input_bytes": self.estimated_input_bytes,
            "estimated_output_bytes": self.estimated_output_bytes,
            "estimated_workspace_bytes": self.estimated_workspace_bytes,
            "estimated_memory_bytes": self.estimated_memory_bytes,
            "workload": self.workload.value,
            "supported_backends": [
                backend.value for backend in self.supported_backends
            ],
            "precision": self.precision.value,
            "parallelizable": self.parallelizable,
            "supports_batching": self.supports_batching,
            "complexity_model": self.complexity_model,
            "assumptions": list(self.assumptions),
        }


class TaskAnalyzer:
    """Оценивает вычислительную сложность задач по их параметрам."""

    BYTES_PER_ELEMENT = {
        PrecisionRequirement.EXACT: 8,
        PrecisionRequirement.INTEGER: 8,
        PrecisionRequirement.FLOAT32: 4,
        PrecisionRequirement.FLOAT64: 8,
    }

    def __init__(
        self,
        registry: TaskRegistry,
        workspace_multiplier: float = 1.5,
    ) -> None:
        if not isinstance(registry, TaskRegistry):
            raise TypeError("registry должен быть TaskRegistry")

        if (
            isinstance(workspace_multiplier, bool)
            or not isinstance(workspace_multiplier, (int, float))
        ):
            raise TypeError("workspace_multiplier должен быть числом")

        if not math.isfinite(workspace_multiplier):
            raise ValueError(
                "workspace_multiplier должен быть конечным числом"
            )

        if workspace_multiplier < 1:
            raise ValueError("workspace_multiplier не может быть меньше 1")

        self.registry = registry
        self.workspace_multiplier = float(workspace_multiplier)

    @staticmethod
    def _required(
        parameters: TaskParameters,
        *names: str,
    ) -> tuple[int, ...]:
        """Получает обязательные параметры или сообщает об ошибке."""
        values = []

        for name in names:
            value = getattr(parameters, name)

            if value is None:
                raise ValueError(
                    f"Для анализа задачи необходимо указать {name}"
                )

            values.append(value)

        return tuple(values)

    @staticmethod
    def _workspace(output_bytes: int, multiplier: float) -> int:
        """Оценивает дополнительную рабочую память."""
        return int(output_bytes * (multiplier - 1))

    def analyze(
        self,
        task_type: TaskType,
        parameters: TaskParameters,
    ) -> TaskAnalysis:
        """Анализирует задачу и оценивает её вычислительные ресурсы."""
        if not isinstance(parameters, TaskParameters):
            raise TypeError("parameters должен быть TaskParameters")

        definition = self.registry.get(task_type)
        element_size = self.BYTES_PER_ELEMENT[definition.precision]

        method = getattr(self, f"_analyze_{task_type.value}")
        metrics = method(parameters, element_size)

        input_bytes = metrics["input_bytes"]
        output_bytes = metrics["output_bytes"]
        workspace_bytes = max(
            metrics["workspace_bytes"],
            self._workspace(output_bytes, self.workspace_multiplier),
        )

        return TaskAnalysis(
            task_type=definition.task_type,
            item_count=metrics["item_count"],
            estimated_operations=metrics["operations"],
            estimated_input_bytes=input_bytes,
            estimated_output_bytes=output_bytes,
            estimated_workspace_bytes=workspace_bytes,
            estimated_memory_bytes=(
                input_bytes + output_bytes + workspace_bytes
            ),
            workload=definition.workload,
            supported_backends=definition.supported_backends,
            precision=definition.precision,
            parallelizable=definition.parallelizable,
            supports_batching=definition.supports_batching,
            complexity_model=metrics["complexity_model"],
            assumptions=metrics["assumptions"],
        )

    def _analyze_matrix_multiplication(
        self,
        p: TaskParameters,
        element_size: int,
    ) -> dict:
        m, k, n = self._required(
            p, "matrix_m", "matrix_k", "matrix_n"
        )

        input_bytes = (m * k + k * n) * element_size
        output_bytes = m * n * element_size

        return {
            "item_count": m * k * n,
            "operations": 2 * m * k * n,
            "input_bytes": input_bytes,
            "output_bytes": output_bytes,
            "workspace_bytes": 0,
            "complexity_model": "O(M*K*N)",
            "assumptions": (
                "Оценка операций учитывает умножение и сложение.",
                "Входные матрицы и результат хранятся одновременно.",
            ),
        }

    def _analyze_statistical_analysis(
        self,
        p: TaskParameters,
        element_size: int,
    ) -> dict:
        samples, features = self._required(
            p, "sample_count", "feature_count"
        )

        input_bytes = samples * features * element_size
        output_bytes = features * element_size

        return {
            "item_count": samples * features,
            "operations": 10 * samples * features,
            "input_bytes": input_bytes,
            "output_bytes": output_bytes,
            "workspace_bytes": output_bytes,
            "complexity_model": "O(samples*features)",
            "assumptions": (
                "Используется условная оценка 10 операций на элемент.",
                "Результат содержит по одному значению на признак.",
            ),
        }

    def _analyze_combination_generation(
        self,
        p: TaskParameters,
        element_size: int,
    ) -> dict:
        candidates, width = self._required(
            p, "candidate_count", "item_width"
        )

        input_bytes = width * element_size
        output_bytes = candidates * width * element_size

        return {
            "item_count": candidates,
            "operations": candidates * width,
            "input_bytes": input_bytes,
            "output_bytes": output_bytes,
            "workspace_bytes": 0,
            "complexity_model": "O(candidates*width)",
            "assumptions": (
                "Количество кандидатов задаётся заранее.",
                "Оценка не включает поиск и перебор кандидатов.",
                "Каждый кандидат содержит width элементов.",
            ),
        }

    def _analyze_monte_carlo(
        self,
        p: TaskParameters,
        element_size: int,
    ) -> dict:
        iterations, operations, state_size = self._required(
            p,
            "iterations",
            "operations_per_iteration",
            "state_size",
        )

        input_bytes = state_size * element_size
        output_bytes = iterations * element_size

        return {
            "item_count": iterations,
            "operations": iterations * operations,
            "input_bytes": input_bytes,
            "output_bytes": output_bytes,
            "workspace_bytes": state_size * element_size,
            "complexity_model": "O(iterations*operations_per_iteration)",
            "assumptions": (
                "Количество операций одной симуляции задаётся пользователем.",
                "Выход содержит одно значение на итерацию.",
                "Рабочее состояние оценивается по state_size.",
            ),
        }

    def _analyze_genetic_optimization(
        self,
        p: TaskParameters,
        element_size: int,
    ) -> dict:
        population, generations, genome = self._required(
            p,
            "population_size",
            "generations",
            "genome_size",
        )

        input_bytes = population * genome * element_size
        output_bytes = population * genome * element_size

        return {
            "item_count": population * generations,
            "operations": population * generations * genome,
            "input_bytes": input_bytes,
            "output_bytes": output_bytes,
            "workspace_bytes": input_bytes,
            "complexity_model": "O(population*generations*genome)",
            "assumptions": (
                "Оценивается обработка каждого гена каждого поколения.",
                "Не учитывается переменная стоимость функции приспособленности.",
                "Размер популяции считается постоянным.",
            ),
        }

    def _analyze_model_training(
        self,
        p: TaskParameters,
        element_size: int,
    ) -> dict:
        samples, features, epochs = self._required(
            p, "sample_count", "feature_count", "epochs"
        )

        batch_size = p.batch_size or samples
        input_bytes = samples * features * element_size
        output_bytes = features * element_size

        return {
            "item_count": samples * epochs,
            "operations": 2 * samples * features * epochs,
            "input_bytes": input_bytes,
            "output_bytes": output_bytes,
            "workspace_bytes": batch_size * features * element_size,
            "complexity_model": "O(samples*features*epochs)",
            "assumptions": (
                "Условно учитываются прямой и обратный проходы.",
                "Сложность архитектуры модели не учитывается.",
                "Память пакета оценивается по batch_size.",
            ),
        }

    def _analyze_hypothesis_testing(
        self,
        p: TaskParameters,
        element_size: int,
    ) -> dict:
        samples, hypotheses = self._required(
            p, "sample_count", "hypothesis_count"
        )

        input_bytes = samples * element_size
        output_bytes = hypotheses * element_size

        return {
            "item_count": hypotheses,
            "operations": samples * hypotheses * 10,
            "input_bytes": input_bytes,
            "output_bytes": output_bytes,
            "workspace_bytes": output_bytes,
            "complexity_model": "O(samples*hypotheses)",
            "assumptions": (
                "Используется условная оценка 10 операций на пару.",
                "Сложность конкретного статистического теста не учитывается.",
            ),
        }
