
"""Тесты типов данных Execution Engine."""

import json

import pytest

from services.compute.scheduler.execution_types import (
    ExecutionErrorCode,
    ExecutionMetrics,
    ExecutionResult,
    ExecutionStatus,
)
from services.compute.scheduler.task_types import (
    ComputeBackend,
    TaskType,
)


def make_metrics():
    return ExecutionMetrics(
        execution_seconds=1.0,
        preparation_seconds=0.1,
        validation_seconds=0.1,
        batch_count=2,
        processed_items=100,
    )


def test_metrics_accept_valid_values():
    metrics = make_metrics()

    assert metrics.batch_count == 2
    assert metrics.processed_items == 100


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("execution_seconds", -1.0, ValueError),
        ("preparation_seconds", float("inf"), ValueError),
        ("validation_seconds", float("nan"), ValueError),
        ("execution_seconds", True, TypeError),
        ("batch_count", True, TypeError),
        ("processed_items", 1.5, TypeError),
        ("batch_count", -1, ValueError),
        ("processed_items", -1, ValueError),
        ("peak_memory_bytes", -1, ValueError),
        ("peak_memory_bytes", True, TypeError),
    ],
)
def test_metrics_validate_fields(field, value, error):
    values = {
        "execution_seconds": 1.0,
        "preparation_seconds": 0.1,
        "validation_seconds": 0.1,
        "batch_count": 2,
        "processed_items": 100,
        "peak_memory_bytes": None,
    }
    values[field] = value

    with pytest.raises(error):
        ExecutionMetrics(**values)


def test_success_result_requires_backend_and_validation():
    with pytest.raises(ValueError):
        ExecutionResult(
            task_type=TaskType.MATRIX_MULTIPLICATION,
            status=ExecutionStatus.SUCCESS,
            requested_backend=ComputeBackend.CUDA,
            actual_backend=None,
            result=[],
            metrics=make_metrics(),
            result_validated=True,
        )


def test_success_result_requires_validated_result():
    with pytest.raises(ValueError):
        ExecutionResult(
            task_type=TaskType.MATRIX_MULTIPLICATION,
            status=ExecutionStatus.SUCCESS,
            requested_backend=ComputeBackend.CUDA,
            actual_backend=ComputeBackend.CUDA,
            result=[],
            metrics=make_metrics(),
            result_validated=False,
        )


def test_failed_result_requires_error_code():
    with pytest.raises(ValueError):
        ExecutionResult(
            task_type=TaskType.MATRIX_MULTIPLICATION,
            status=ExecutionStatus.FAILED,
            requested_backend=ComputeBackend.CUDA,
            actual_backend=None,
            result=None,
            metrics=make_metrics(),
        )


def test_cancelled_result_requires_cancelled_error_code():
    with pytest.raises(ValueError):
        ExecutionResult(
            task_type=TaskType.MATRIX_MULTIPLICATION,
            status=ExecutionStatus.CANCELLED,
            requested_backend=ComputeBackend.CUDA,
            actual_backend=None,
            result=None,
            metrics=make_metrics(),
            error_code=ExecutionErrorCode.EXECUTION_ERROR,
        )


def test_result_can_be_serialized_to_json():
    result = ExecutionResult(
        task_type=TaskType.MATRIX_MULTIPLICATION,
        status=ExecutionStatus.SUCCESS,
        requested_backend=ComputeBackend.CUDA,
        actual_backend=ComputeBackend.CUDA,
        result=[[1, 2], [3, 4]],
        metrics=make_metrics(),
        result_validated=True,
    )

    serialized = json.dumps(result.as_dict())

    assert "matrix_multiplication" in serialized
    assert "success" in serialized
    assert "cuda" in serialized
