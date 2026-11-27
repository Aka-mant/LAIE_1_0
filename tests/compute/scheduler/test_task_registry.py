
"""Тесты реестра вычислительных задач LAIE."""

import pytest

from services.compute.scheduler import (
    ComputeBackend,
    PrecisionRequirement,
    TaskAlreadyRegisteredError,
    TaskDefinition,
    TaskNotFoundError,
    TaskRegistry,
    TaskType,
    WorkloadKind,
    create_default_registry,
)


def make_definition(
    task_type=TaskType.MATRIX_MULTIPLICATION,
):
    """Создаёт тестовое описание вычислительной задачи."""
    return TaskDefinition(
        task_type=task_type,
        name="Тестовая задача",
        description="Тестовое описание.",
        workload=WorkloadKind.CPU_BOUND,
        supported_backends=(ComputeBackend.CPU,),
        precision=PrecisionRequirement.EXACT,
        parallelizable=True,
        supports_batching=False,
        estimator_key="test",
    )


def test_empty_registry_has_zero_tasks():
    registry = TaskRegistry()

    assert len(registry) == 0
    assert registry.list_all() == ()
    assert registry.list_types() == ()


def test_register_adds_task():
    registry = TaskRegistry()
    definition = make_definition()

    registry.register(definition)

    assert len(registry) == 1
    assert registry.contains(definition.task_type)


def test_get_returns_registered_definition():
    registry = TaskRegistry()
    definition = make_definition()

    registry.register(definition)

    assert registry.get(definition.task_type) is definition


def test_duplicate_registration_is_rejected():
    registry = TaskRegistry()
    definition = make_definition()

    registry.register(definition)

    with pytest.raises(TaskAlreadyRegisteredError):
        registry.register(definition)


def test_unknown_task_raises_error():
    registry = TaskRegistry()

    with pytest.raises(TaskNotFoundError):
        registry.get(TaskType.MONTE_CARLO)


def test_contains_returns_false_for_unknown_task():
    registry = TaskRegistry()

    assert not registry.contains(TaskType.MONTE_CARLO)


def test_contains_operator():
    registry = TaskRegistry()
    definition = make_definition()

    registry.register(definition)

    assert definition.task_type in registry
    assert TaskType.MONTE_CARLO not in registry


def test_register_rejects_invalid_object():
    registry = TaskRegistry()

    with pytest.raises(TypeError):
        registry.register("invalid")


@pytest.mark.parametrize(
    "method",
    ["get", "contains"],
)
def test_methods_reject_invalid_task_type(method):
    registry = TaskRegistry()

    with pytest.raises(TypeError):
        getattr(registry, method)("matrix_multiplication")


def test_list_all_returns_tuple():
    registry = create_default_registry()

    result = registry.list_all()

    assert isinstance(result, tuple)
    assert len(result) == 7


def test_list_types_returns_tuple():
    registry = create_default_registry()

    result = registry.list_types()

    assert isinstance(result, tuple)
    assert all(isinstance(item, TaskType) for item in result)


def test_default_registry_contains_all_task_types():
    registry = create_default_registry()

    assert set(registry.list_types()) == set(TaskType)


def test_default_registry_has_unique_estimator_keys():
    registry = create_default_registry()

    keys = [
        definition.estimator_key
        for definition in registry.list_all()
    ]

    assert len(keys) == len(set(keys))


def test_default_registry_has_cpu_fallback_for_parallel_tasks():
    registry = create_default_registry()

    for definition in registry.list_all():
        if ComputeBackend.CUDA in definition.supported_backends:
            assert ComputeBackend.CPU in definition.supported_backends


def test_default_registry_returns_expected_matrix_definition():
    registry = create_default_registry()

    definition = registry.get(TaskType.MATRIX_MULTIPLICATION)

    assert definition.name == "Матричные операции"
    assert definition.workload == WorkloadKind.GPU_PARALLEL
    assert definition.supports_backend(ComputeBackend.CUDA)


def test_separate_registries_are_independent():
    first = create_default_registry()
    second = TaskRegistry()

    assert len(first) == 7
    assert len(second) == 0


def test_registration_order_is_preserved():
    registry = create_default_registry()

    types = registry.list_types()

    assert types[0] == TaskType.MATRIX_MULTIPLICATION
    assert types[-1] == TaskType.HYPOTHESIS_TESTING
