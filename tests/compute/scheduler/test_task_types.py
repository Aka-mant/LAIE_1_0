
"""Тесты контрактов вычислительных задач LAIE."""

import pytest

from services.compute.scheduler import (
    ComputeBackend,
    PrecisionRequirement,
    TaskDefinition,
    TaskType,
    WorkloadKind,
)


def make_definition(**overrides):
    """Создаёт корректное описание задачи для тестирования."""
    values = {
        "task_type": TaskType.MATRIX_MULTIPLICATION,
        "name": "Тестовая задача",
        "description": "Тестовое описание.",
        "workload": WorkloadKind.GPU_PARALLEL,
        "supported_backends": (
            ComputeBackend.CPU,
            ComputeBackend.CUDA,
        ),
        "precision": PrecisionRequirement.FLOAT64,
        "parallelizable": True,
        "supports_batching": True,
        "estimator_key": "test",
    }
    values.update(overrides)
    return TaskDefinition(**values)


def test_valid_definition_is_created():
    definition = make_definition()

    assert definition.task_type == TaskType.MATRIX_MULTIPLICATION
    assert definition.name == "Тестовая задача"
    assert definition.parallelizable is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", ""),
        ("name", "   "),
        ("description", ""),
        ("description", "   "),
        ("estimator_key", ""),
        ("estimator_key", "   "),
    ],
)
def test_empty_required_text_is_rejected(field, value):
    with pytest.raises(ValueError):
        make_definition(**{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("task_type", "matrix_multiplication"),
        ("workload", "gpu_parallel"),
        ("precision", "float64"),
    ],
)
def test_invalid_enum_fields_are_rejected(field, value):
    with pytest.raises(TypeError):
        make_definition(**{field: value})


def test_empty_backend_list_is_rejected():
    with pytest.raises(ValueError):
        make_definition(supported_backends=())


def test_duplicate_backends_are_rejected():
    with pytest.raises(ValueError):
        make_definition(
            supported_backends=(
                ComputeBackend.CPU,
                ComputeBackend.CPU,
            )
        )


def test_invalid_backend_is_rejected():
    with pytest.raises(TypeError):
        make_definition(
            supported_backends=(
                ComputeBackend.CPU,
                "cuda",
            )
        )


@pytest.mark.parametrize(
    "field",
    ["parallelizable", "supports_batching"],
)
def test_boolean_fields_are_validated(field):
    with pytest.raises(TypeError):
        make_definition(**{field: 1})


def test_supports_backend_returns_true_for_supported_device():
    definition = make_definition()

    assert definition.supports_backend(ComputeBackend.CPU)
    assert definition.supports_backend(ComputeBackend.CUDA)


def test_supports_backend_returns_false_for_unsupported_device():
    definition = make_definition(
        supported_backends=(ComputeBackend.CPU,)
    )

    assert not definition.supports_backend(ComputeBackend.CUDA)


def test_definition_is_immutable():
    definition = make_definition()

    with pytest.raises(AttributeError):
        definition.name = "Изменённое название"
