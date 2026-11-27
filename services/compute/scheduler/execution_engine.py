
"""Локальный исполнитель вычислительных задач с безопасным fallback."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import Event
from time import perf_counter
from typing import Any

from .execution_types import (
    ExecutionErrorCode,
    ExecutionMetrics,
    ExecutionResult,
    ExecutionStatus,
)
from .task_types import ComputeBackend, TaskType


BatchExecutor = Callable[
    [int, int, ComputeBackend],
    Any,
]

ResultCombiner = Callable[[list[Any]], Any]
ResultValidator = Callable[[Any, int], bool]
MemoryGuard = Callable[[int, int, ComputeBackend], bool]


class ExecutionCancelledError(Exception):
    """Задача отменена до завершения."""


class ExecutionMemoryError(Exception):
    """Недостаточно памяти для выполнения пакета."""


@dataclass(frozen=True, slots=True)
class ExecutionRequest:
    """Параметры запуска вычислительной задачи."""

    task_type: TaskType
    item_count: int
    batch_size: int
    requested_backend: ComputeBackend
    retry_safe: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.task_type, TaskType):
            raise TypeError("task_type должен быть TaskType.")

        if not isinstance(self.requested_backend, ComputeBackend):
            raise TypeError(
                "requested_backend должен быть ComputeBackend."
            )

        for name in ("item_count", "batch_size"):
            value = getattr(self, name)

            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} должен быть целым числом.")

            if value <= 0:
                raise ValueError(f"{name} должен быть больше нуля.")

        if not isinstance(self.retry_safe, bool):
            raise TypeError("retry_safe должен быть bool.")


class ExecutionEngine:
    """Исполняет задачи пакетами и выполняет безопасный fallback."""

    def __init__(
        self,
        *,
        cpu_fallback_enabled: bool = True,
        memory_guard: MemoryGuard | None = None,
    ) -> None:
        if not isinstance(cpu_fallback_enabled, bool):
            raise TypeError("cpu_fallback_enabled должен быть bool.")

        if memory_guard is not None and not callable(memory_guard):
            raise TypeError("memory_guard должен быть callable.")

        self.cpu_fallback_enabled = cpu_fallback_enabled
        self.memory_guard = memory_guard

    def execute(
            self,
            request: ExecutionRequest,
            batch_executor: BatchExecutor,
            *,
            result_combiner: ResultCombiner | None = None,
            result_validator: ResultValidator | None = None,
            cancellation_event: Event | None = None,
            memory_guard: MemoryGuard | None = None,
    ) -> ExecutionResult:
        """Выполняет задачу и при необходимости повторяет её на CPU."""
        if not isinstance(request, ExecutionRequest):
            raise TypeError("request должен быть ExecutionRequest.")

        if not callable(batch_executor):
            raise TypeError("batch_executor должен быть callable.")

        if result_combiner is not None and not callable(result_combiner):
            raise TypeError("result_combiner должен быть callable.")

        if result_validator is not None and not callable(result_validator):
            raise TypeError("result_validator должен быть callable.")

        if cancellation_event is not None and not isinstance(
            cancellation_event,
            Event,
        ):
            raise TypeError("cancellation_event должен быть threading.Event.")
        if memory_guard is not None and not callable(memory_guard):
            raise TypeError("memory_guard должен быть callable.")

        active_memory_guard = (
            memory_guard
            if memory_guard is not None
            else self.memory_guard
        )

        first = self._execute_backend(
            request,
            request.requested_backend,
            batch_executor,
            result_combiner,
            result_validator,
            cancellation_event,
            active_memory_guard,
        )

        if first.status == ExecutionStatus.SUCCESS:
            return first

        if first.status == ExecutionStatus.CANCELLED:
            return first

        if (
            request.requested_backend != ComputeBackend.CUDA
            or not self.cpu_fallback_enabled
            or not request.retry_safe
        ):
            return first

        fallback = self._execute_backend(
            request,
            ComputeBackend.CPU,
            batch_executor,
            result_combiner,
            result_validator,
            cancellation_event,
            active_memory_guard,
        )


        combined_metrics = ExecutionMetrics(
            execution_seconds=(
                first.metrics.execution_seconds
                + fallback.metrics.execution_seconds
            ),
            preparation_seconds=(
                first.metrics.preparation_seconds
                + fallback.metrics.preparation_seconds
            ),
            validation_seconds=(
                first.metrics.validation_seconds
                + fallback.metrics.validation_seconds
            ),
            batch_count=(
                first.metrics.batch_count
                + fallback.metrics.batch_count
            ),
            processed_items=fallback.metrics.processed_items,
            peak_memory_bytes=fallback.metrics.peak_memory_bytes,
        )

        if fallback.status == ExecutionStatus.SUCCESS:
            return ExecutionResult(
                task_type=request.task_type,
                status=ExecutionStatus.SUCCESS,
                requested_backend=request.requested_backend,
                actual_backend=ComputeBackend.CPU,
                result=fallback.result,
                metrics=combined_metrics,
                fallback_used=True,
                result_validated=True,
            )

        return ExecutionResult(
            task_type=request.task_type,
            status=fallback.status,
            requested_backend=request.requested_backend,
            actual_backend=None,
            result=None,
            metrics=combined_metrics,
            error_code=fallback.error_code,
            error_message=(
                "CUDA завершилась ошибкой: "
                f"{first.error_message}; CPU fallback: "
                f"{fallback.error_message}"
            ),
            fallback_used=True,
            result_validated=False,
        )

    def _execute_backend(
            self,
            request: ExecutionRequest,
            backend: ComputeBackend,
            batch_executor: BatchExecutor,
            result_combiner: ResultCombiner | None,
            result_validator: ResultValidator | None,
            cancellation_event: Event | None,
            memory_guard: MemoryGuard | None,
    ) -> ExecutionResult:
        """Выполняет все пакеты на одном backend."""
        started = perf_counter()
        batch_results: list[Any] = []
        batch_count = 0
        processed_items = 0
        preparation_seconds = 0.0
        validation_seconds = 0.0

        try:
            for start in range(0, request.item_count, request.batch_size):
                self._check_cancelled(cancellation_event)

                stop = min(
                    start + request.batch_size,
                    request.item_count,
                )
                current_batch_size = stop - start

                if memory_guard is not None:
                    preparation_started = perf_counter()
                    fits = memory_guard(
                        start,
                        stop,
                        backend,
                    )
                    preparation_seconds += (
                        perf_counter() - preparation_started
                    )

                    if not isinstance(fits, bool):
                        raise TypeError(
                            "memory_guard должен возвращать bool."
                        )

                    if not fits:
                        raise ExecutionMemoryError(
                            f"Пакет {start}:{stop} не помещается в памяти."
                        )

                self._check_cancelled(cancellation_event)

                batch_result = batch_executor(
                    start,
                    stop,
                    backend,
                )

                batch_results.append(batch_result)
                batch_count += 1
                processed_items += current_batch_size

            self._check_cancelled(cancellation_event)

            preparation_started = perf_counter()

            if result_combiner is None:
                combined_result = batch_results
            else:
                combined_result = result_combiner(batch_results)

            preparation_seconds += perf_counter() - preparation_started

            validation_started = perf_counter()

            if result_validator is None:
                valid = (
                        combined_result is not None
                        and all(item is not None for item in batch_results)
                )
            else:
                valid = result_validator(
                    combined_result,
                    request.item_count,
                )

            validation_seconds = perf_counter() - validation_started

            if not isinstance(valid, bool):
                raise TypeError(
                    "result_validator должен возвращать bool."
                )

            if not valid:
                raise ValueError(
                    "Результат не прошёл проверку целостности."
                )

            elapsed = perf_counter() - started

            return ExecutionResult(
                task_type=request.task_type,
                status=ExecutionStatus.SUCCESS,
                requested_backend=request.requested_backend,
                actual_backend=backend,
                result=combined_result,
                metrics=ExecutionMetrics(
                    execution_seconds=max(
                        0.0,
                        elapsed
                        - preparation_seconds
                        - validation_seconds,
                    ),
                    preparation_seconds=preparation_seconds,
                    validation_seconds=validation_seconds,
                    batch_count=batch_count,
                    processed_items=processed_items,
                ),
                result_validated=True,
            )

        except ExecutionCancelledError as exc:
            status = ExecutionStatus.CANCELLED
            error_code = ExecutionErrorCode.CANCELLED
            error_message = str(exc)

        except ExecutionMemoryError as exc:
            status = ExecutionStatus.FAILED
            error_code = ExecutionErrorCode.MEMORY_ERROR
            error_message = str(exc)

        except MemoryError as exc:
            status = ExecutionStatus.FAILED
            error_code = ExecutionErrorCode.MEMORY_ERROR
            error_message = str(exc) or "Недостаточно оперативной памяти."

        except (RuntimeError, OSError) as exc:
            status = ExecutionStatus.FAILED
            error_code = ExecutionErrorCode.EXECUTION_ERROR
            error_message = f"{type(exc).__name__}: {exc}"

        except ValueError as exc:
            status = ExecutionStatus.FAILED
            error_code = ExecutionErrorCode.VALIDATION_ERROR
            error_message = str(exc)

        except Exception as exc:
            status = ExecutionStatus.FAILED
            error_code = ExecutionErrorCode.EXECUTION_ERROR
            error_message = f"{type(exc).__name__}: {exc}"

        elapsed = perf_counter() - started

        return ExecutionResult(
            task_type=request.task_type,
            status=status,
            requested_backend=request.requested_backend,
            actual_backend=None,
            result=None,
            metrics=ExecutionMetrics(
                execution_seconds=max(
                    0.0,
                    elapsed - preparation_seconds - validation_seconds,
                ),
                preparation_seconds=preparation_seconds,
                validation_seconds=validation_seconds,
                batch_count=batch_count,
                processed_items=processed_items,
            ),
            error_code=error_code,
            error_message=error_message,
            result_validated=False,
        )

    @staticmethod
    def _check_cancelled(
        cancellation_event: Event | None,
    ) -> None:
        if cancellation_event is not None and cancellation_event.is_set():
            raise ExecutionCancelledError(
                "Выполнение отменено пользователем."
            )
