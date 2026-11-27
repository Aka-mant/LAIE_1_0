
"""Тесты локального Execution Engine."""

from threading import Event

import pytest

from services.compute.scheduler.execution_engine import (
    ExecutionEngine,
    ExecutionRequest,
)
from services.compute.scheduler.execution_types import (
    ExecutionErrorCode,
    ExecutionStatus,
)
from services.compute.scheduler.task_types import (
    ComputeBackend,
    TaskType,
)


@pytest.fixture
def request_cuda():
    return ExecutionRequest(
        task_type=TaskType.MATRIX_MULTIPLICATION,
        item_count=10,
        batch_size=3,
        requested_backend=ComputeBackend.CUDA,
        retry_safe=True,
    )


@pytest.fixture
def engine():
    return ExecutionEngine()


def test_request_validates_item_count():
    with pytest.raises(ValueError):
        ExecutionRequest(
            task_type=TaskType.MATRIX_MULTIPLICATION,
            item_count=0,
            batch_size=1,
            requested_backend=ComputeBackend.CPU,
        )


def test_request_rejects_boolean_batch_size():
    with pytest.raises(TypeError):
        ExecutionRequest(
            task_type=TaskType.MATRIX_MULTIPLICATION,
            item_count=10,
            batch_size=True,
            requested_backend=ComputeBackend.CPU,
        )


def test_request_rejects_invalid_retry_flag():
    with pytest.raises(TypeError):
        ExecutionRequest(
            task_type=TaskType.MATRIX_MULTIPLICATION,
            item_count=10,
            batch_size=2,
            requested_backend=ComputeBackend.CPU,
            retry_safe=1,
        )


def test_cpu_execution_processes_all_batches():
    calls = []

    def execute_batch(start, stop, backend):
        calls.append((start, stop, backend))
        return list(range(start, stop))

    result = ExecutionEngine().execute(
        ExecutionRequest(
            task_type=TaskType.MATRIX_MULTIPLICATION,
            item_count=10,
            batch_size=3,
            requested_backend=ComputeBackend.CPU,
        ),
        execute_batch,
        result_combiner=lambda batches: [
            item
            for batch in batches
            for item in batch
        ],
        result_validator=lambda value, expected: len(value) == expected,
    )

    assert result.status == ExecutionStatus.SUCCESS
    assert result.actual_backend == ComputeBackend.CPU
    assert result.metrics.batch_count == 4
    assert result.metrics.processed_items == 10
    assert result.result == list(range(10))
    assert len(calls) == 4


def test_last_batch_is_smaller():
    batch_sizes = []

    def execute_batch(start, stop, backend):
        batch_sizes.append(stop - start)
        return stop - start

    result = ExecutionEngine().execute(
        ExecutionRequest(
            task_type=TaskType.STATISTICAL_ANALYSIS,
            item_count=10,
            batch_size=4,
            requested_backend=ComputeBackend.CPU,
        ),
        execute_batch,
    )

    assert result.status == ExecutionStatus.SUCCESS
    assert batch_sizes == [4, 4, 2]


def test_cuda_failure_falls_back_to_cpu_when_retry_is_safe():
    calls = []

    def execute_batch(start, stop, backend):
        calls.append(backend)

        if backend == ComputeBackend.CUDA:
            raise RuntimeError("CUDA out of memory")

        return list(range(start, stop))

    result = ExecutionEngine().execute(
        ExecutionRequest(
            task_type=TaskType.MATRIX_MULTIPLICATION,
            item_count=4,
            batch_size=2,
            requested_backend=ComputeBackend.CUDA,
            retry_safe=True,
        ),
        execute_batch,
        result_combiner=lambda batches: [
            item
            for batch in batches
            for item in batch
        ],
        result_validator=lambda value, expected: len(value) == expected,
    )

    assert result.status == ExecutionStatus.SUCCESS
    assert result.requested_backend == ComputeBackend.CUDA
    assert result.actual_backend == ComputeBackend.CPU
    assert result.fallback_used is True
    assert result.result == [0, 1, 2, 3]
    assert calls == [
        ComputeBackend.CUDA,
        ComputeBackend.CPU,
        ComputeBackend.CPU,
    ]


def test_cuda_failure_does_not_retry_when_not_safe():
    calls = []

    def execute_batch(start, stop, backend):
        calls.append(backend)
        raise RuntimeError("GPU error")

    result = ExecutionEngine().execute(
        ExecutionRequest(
            task_type=TaskType.MATRIX_MULTIPLICATION,
            item_count=4,
            batch_size=2,
            requested_backend=ComputeBackend.CUDA,
            retry_safe=False,
        ),
        execute_batch,
    )

    assert result.status == ExecutionStatus.FAILED
    assert result.fallback_used is False
    assert calls == [ComputeBackend.CUDA]


def test_cpu_fallback_can_be_disabled():
    calls = []

    def execute_batch(start, stop, backend):
        calls.append(backend)
        raise RuntimeError("GPU error")

    result = ExecutionEngine(
        cpu_fallback_enabled=False,
    ).execute(
        ExecutionRequest(
            task_type=TaskType.MATRIX_MULTIPLICATION,
            item_count=2,
            batch_size=2,
            requested_backend=ComputeBackend.CUDA,
            retry_safe=True,
        ),
        execute_batch,
    )

    assert result.status == ExecutionStatus.FAILED
    assert calls == [ComputeBackend.CUDA]


def test_memory_guard_prevents_batch_execution():
    calls = []

    def memory_guard(start, stop, backend):
        return False

    def execute_batch(start, stop, backend):
        calls.append((start, stop))
        return [1]

    result = ExecutionEngine(
        memory_guard=memory_guard,
    ).execute(
        ExecutionRequest(
            task_type=TaskType.MATRIX_MULTIPLICATION,
            item_count=4,
            batch_size=2,
            requested_backend=ComputeBackend.CPU,
        ),
        execute_batch,
    )

    assert result.status == ExecutionStatus.FAILED
    assert result.error_code == ExecutionErrorCode.MEMORY_ERROR
    assert calls == []


def test_memory_guard_is_checked_for_every_batch():
    checks = []

    def memory_guard(start, stop, backend):
        checks.append((start, stop))
        return True

    result = ExecutionEngine(
        memory_guard=memory_guard,
    ).execute(
        ExecutionRequest(
            task_type=TaskType.STATISTICAL_ANALYSIS,
            item_count=5,
            batch_size=2,
            requested_backend=ComputeBackend.CPU,
        ),
        lambda start, stop, backend: stop - start,
    )

    assert result.status == ExecutionStatus.SUCCESS
    assert checks == [(0, 2), (2, 4), (4, 5)]


def test_result_validation_failure_is_reported():
    result = ExecutionEngine().execute(
        ExecutionRequest(
            task_type=TaskType.STATISTICAL_ANALYSIS,
            item_count=5,
            batch_size=2,
            requested_backend=ComputeBackend.CPU,
        ),
        lambda start, stop, backend: [1],
        result_validator=lambda value, expected: False,
    )

    assert result.status == ExecutionStatus.FAILED
    assert result.error_code == ExecutionErrorCode.VALIDATION_ERROR
    assert result.result is None


def test_empty_result_is_rejected_by_default():
    result = ExecutionEngine().execute(
        ExecutionRequest(
            task_type=TaskType.STATISTICAL_ANALYSIS,
            item_count=2,
            batch_size=2,
            requested_backend=ComputeBackend.CPU,
        ),
        lambda start, stop, backend: None,
    )

    assert result.status == ExecutionStatus.FAILED
    assert result.error_code == ExecutionErrorCode.VALIDATION_ERROR


def test_cancellation_before_execution():
    cancellation = Event()
    cancellation.set()
    calls = []

    result = ExecutionEngine().execute(
        ExecutionRequest(
            task_type=TaskType.STATISTICAL_ANALYSIS,
            item_count=5,
            batch_size=2,
            requested_backend=ComputeBackend.CPU,
        ),
        lambda start, stop, backend: calls.append(start),
        cancellation_event=cancellation,
    )

    assert result.status == ExecutionStatus.CANCELLED
    assert result.error_code == ExecutionErrorCode.CANCELLED
    assert calls == []


def test_cancellation_between_batches():
    cancellation = Event()
    calls = []

    def execute_batch(start, stop, backend):
        calls.append(start)

        if start == 0:
            cancellation.set()

        return [1]

    result = ExecutionEngine().execute(
        ExecutionRequest(
            task_type=TaskType.STATISTICAL_ANALYSIS,
            item_count=5,
            batch_size=2,
            requested_backend=ComputeBackend.CPU,
        ),
        execute_batch,
        cancellation_event=cancellation,
    )

    assert result.status == ExecutionStatus.CANCELLED
    assert calls == [0]


def test_memory_error_triggers_safe_fallback():
    calls = []

    def execute_batch(start, stop, backend):
        calls.append(backend)

        if backend == ComputeBackend.CUDA:
            raise MemoryError("GPU memory exhausted")

        return [1] * (stop - start)

    result = ExecutionEngine().execute(
        ExecutionRequest(
            task_type=TaskType.MATRIX_MULTIPLICATION,
            item_count=4,
            batch_size=2,
            requested_backend=ComputeBackend.CUDA,
            retry_safe=True,
        ),
        execute_batch,
        result_validator=lambda value, expected: (
            sum(len(batch) for batch in value) == expected
        ),
    )

    assert result.status == ExecutionStatus.SUCCESS
    assert result.actual_backend == ComputeBackend.CPU
    assert result.fallback_used is True
    assert calls[0] == ComputeBackend.CUDA
    assert all(item == ComputeBackend.CPU for item in calls[1:])


def test_fallback_failure_is_reported():
    def execute_batch(start, stop, backend):
        raise RuntimeError(f"{backend.value} failed")

    result = ExecutionEngine().execute(
        ExecutionRequest(
            task_type=TaskType.MATRIX_MULTIPLICATION,
            item_count=2,
            batch_size=2,
            requested_backend=ComputeBackend.CUDA,
            retry_safe=True,
        ),
        execute_batch,
    )

    assert result.status == ExecutionStatus.FAILED
    assert result.fallback_used is True
    assert result.actual_backend is None
    assert "CPU fallback" in result.error_message


def test_memory_guard_must_return_boolean():
    result = ExecutionEngine(
        memory_guard=lambda start, stop, backend: 1,
    ).execute(
        ExecutionRequest(
            task_type=TaskType.STATISTICAL_ANALYSIS,
            item_count=2,
            batch_size=2,
            requested_backend=ComputeBackend.CPU,
        ),
        lambda start, stop, backend: [1, 2],
    )

    assert result.status == ExecutionStatus.FAILED
    assert result.error_code == ExecutionErrorCode.EXECUTION_ERROR


def test_invalid_callbacks_are_rejected():
    request = ExecutionRequest(
        task_type=TaskType.STATISTICAL_ANALYSIS,
        item_count=2,
        batch_size=2,
        requested_backend=ComputeBackend.CPU,
    )

    with pytest.raises(TypeError):
        ExecutionEngine().execute(request, None)


def test_invalid_cancellation_object_is_rejected():
    request = ExecutionRequest(
        task_type=TaskType.STATISTICAL_ANALYSIS,
        item_count=2,
        batch_size=2,
        requested_backend=ComputeBackend.CPU,
    )

    with pytest.raises(TypeError):
        ExecutionEngine().execute(
            request,
            lambda start, stop, backend: [1, 2],
            cancellation_event=object(),
        )


def test_failed_cuda_attempt_and_cpu_retry_metrics_are_combined():
    attempts = 0

    def execute_batch(start, stop, backend):
        nonlocal attempts
        attempts += 1

        if backend == ComputeBackend.CUDA:
            raise RuntimeError("CUDA failed")

        return [1] * (stop - start)

    result = ExecutionEngine().execute(
        ExecutionRequest(
            task_type=TaskType.MATRIX_MULTIPLICATION,
            item_count=5,
            batch_size=2,
            requested_backend=ComputeBackend.CUDA,
            retry_safe=True,
        ),
        execute_batch,
        result_validator=lambda value, expected: (
            sum(len(batch) for batch in value) == expected
        ),
    )

    assert result.status == ExecutionStatus.SUCCESS
    assert result.metrics.batch_count == 3
    assert result.metrics.processed_items == 5
    assert attempts == 4
