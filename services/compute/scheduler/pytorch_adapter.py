
"""Адаптер реальных вычислений PyTorch для CPU и CUDA."""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass
from typing import Any

from .task_types import ComputeBackend


class PyTorchAdapterError(RuntimeError):
    """Базовая ошибка PyTorch-адаптера."""


class BackendUnavailableError(PyTorchAdapterError):
    """Запрошенное вычислительное устройство недоступно."""


class InvalidTensorError(PyTorchAdapterError):
    """Входные тензоры не соответствуют контракту адаптера."""


class CUDAExecutionError(PyTorchAdapterError):
    """Ошибка выполнения операции на CUDA."""


@dataclass(frozen=True, slots=True)
class MatmulExecution:
    """Результат матричного умножения и измерения времени."""

    result: Any
    backend: ComputeBackend
    compute_seconds: float
    transfer_seconds: float
    total_seconds: float
    peak_memory_bytes: int | None
    operations: int
    input_bytes: int
    output_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.backend, ComputeBackend):
            raise TypeError("backend должен быть ComputeBackend.")

        for name in (
            "compute_seconds",
            "transfer_seconds",
            "total_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} должен быть числом.")
            if value < 0:
                raise ValueError(f"{name} не может быть отрицательным.")

        if self.compute_seconds <= 0:
            raise ValueError("compute_seconds должен быть больше нуля.")

        if self.total_seconds + 1e-12 < (
            self.compute_seconds + self.transfer_seconds
        ):
            raise ValueError(
                "total_seconds меньше суммы compute_seconds и transfer_seconds."
            )

        if self.backend is ComputeBackend.CPU and self.transfer_seconds != 0:
            raise ValueError("Для CPU transfer_seconds должен быть равен нулю.")

        for name in ("operations", "input_bytes", "output_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} должен быть целым числом.")
            if value < 0:
                raise ValueError(f"{name} не может быть отрицательным.")

        if self.operations <= 0:
            raise ValueError("operations должен быть больше нуля.")

        if self.peak_memory_bytes is not None:
            if self.peak_memory_bytes < 0:
                raise ValueError("peak_memory_bytes не может быть отрицательным.")

    def as_dict(self) -> dict[str, Any]:
        """Возвращает сериализуемые метрики без самого тензора."""
        return {
            "backend": self.backend.value,
            "compute_seconds": self.compute_seconds,
            "transfer_seconds": self.transfer_seconds,
            "total_seconds": self.total_seconds,
            "peak_memory_bytes": self.peak_memory_bytes,
            "operations": self.operations,
            "input_bytes": self.input_bytes,
            "output_bytes": self.output_bytes,
        }


class PyTorchCUDAAdapter:
    """Выполняет операции PyTorch на CPU или CUDA.

    Адаптер не выполняет автоматический fallback. Если CUDA завершилась
    ошибкой, исключение передаётся вызывающему коду. Политика безопасного
    повтора должна определяться ExecutionEngine через retry_safe.
    """

    def __init__(self, torch_module: Any | None = None) -> None:
        self._torch_module = torch_module

    def is_cuda_available(self) -> bool:
        """Проверяет доступность CUDA."""
        try:
            return bool(self._load_torch().cuda.is_available())
        except (ImportError, OSError, RuntimeError, AttributeError):
            return False

    def cuda_device_name(self) -> str | None:
        """Возвращает имя текущего CUDA-устройства."""
        if not self.is_cuda_available():
            return None

        torch = self._load_torch()
        index = torch.cuda.current_device()
        return str(torch.cuda.get_device_properties(index).name)

    def matmul(
        self,
        left: Any,
        right: Any,
        backend: ComputeBackend = ComputeBackend.CPU,
    ) -> MatmulExecution:
        """Умножает две двумерные матрицы.

        Поддерживаются только CPU-тензоры одинакового dtype.
        CUDA-результат копируется обратно на CPU.

        Измерения:
        - compute_seconds: только вычисление;
        - transfer_seconds: копирование CPU→CUDA и CUDA→CPU;
        - total_seconds: вся операция внутри адаптера.
        """
        if not isinstance(backend, ComputeBackend):
            raise TypeError("backend должен быть ComputeBackend.")

        torch = self._load_torch()
        self._validate_inputs(torch, left, right)

        if backend is ComputeBackend.CPU:
            return self._matmul_cpu(torch, left, right)

        if backend is ComputeBackend.CUDA:
            return self._matmul_cuda(torch, left, right)

        raise ValueError(f"Неподдерживаемый backend: {backend!r}")

    @staticmethod
    def _validate_inputs(torch: Any, left: Any, right: Any) -> None:
        """Проверяет форму, тип и устройство входных матриц."""
        if not isinstance(left, torch.Tensor):
            raise InvalidTensorError("left должен быть torch.Tensor.")

        if not isinstance(right, torch.Tensor):
            raise InvalidTensorError("right должен быть torch.Tensor.")

        if left.device.type != "cpu" or right.device.type != "cpu":
            raise InvalidTensorError(
                "Входные тензоры должны находиться на CPU."
            )

        if left.ndim != 2 or right.ndim != 2:
            raise InvalidTensorError(
                "Оба входных тензора должны быть двумерными матрицами."
            )

        if left.shape[1] != right.shape[0]:
            raise InvalidTensorError(
                "Количество столбцов left должно совпадать "
                "с количеством строк right."
            )

        if left.dtype != right.dtype:
            raise InvalidTensorError(
                "Входные матрицы должны иметь одинаковый dtype."
            )

        if left.dtype not in (torch.float32, torch.float64):
            raise InvalidTensorError(
                "Поддерживаются только torch.float32 и torch.float64."
            )

    def _matmul_cpu(
        self,
        torch: Any,
        left: Any,
        right: Any,
    ) -> MatmulExecution:
        """Выполняет матричное умножение на CPU."""
        started = time.perf_counter()

        try:
            result = torch.matmul(left, right)
        except (RuntimeError, OSError) as exc:
            raise PyTorchAdapterError(
                f"Ошибка матричного умножения на CPU: {exc}"
            ) from exc

        total_seconds = time.perf_counter() - started

        if total_seconds <= 0:
            total_seconds = 1e-12

        operations = self._matmul_operations(left, right)
        input_bytes = self._tensor_bytes(left) + self._tensor_bytes(right)
        output_bytes = self._tensor_bytes(result)

        return MatmulExecution(
            result=result,
            backend=ComputeBackend.CPU,
            compute_seconds=total_seconds,
            transfer_seconds=0.0,
            total_seconds=total_seconds,
            peak_memory_bytes=None,
            operations=operations,
            input_bytes=input_bytes,
            output_bytes=output_bytes,
        )

    def _matmul_cuda(
        self,
        torch: Any,
        left: Any,
        right: Any,
    ) -> MatmulExecution:
        """Выполняет умножение на CUDA с синхронизированными замерами."""
        if not torch.cuda.is_available():
            raise BackendUnavailableError(
                "CUDA недоступна на этом компьютере."
            )

        device_index = torch.cuda.current_device()
        device = torch.device(f"cuda:{device_index}")

        total_started = time.perf_counter()

        try:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

            transfer_started = time.perf_counter()

            left_cuda = left.to(device)
            right_cuda = right.to(device)

            torch.cuda.synchronize(device)
            h2d_seconds = time.perf_counter() - transfer_started

            compute_started = time.perf_counter()

            result_cuda = torch.matmul(left_cuda, right_cuda)

            torch.cuda.synchronize(device)
            compute_seconds = time.perf_counter() - compute_started

            transfer_started = time.perf_counter()
            result_cpu = result_cuda.to("cpu")
            torch.cuda.synchronize(device)
            d2h_seconds = time.perf_counter() - transfer_started

            peak_memory = int(
                torch.cuda.max_memory_allocated(device)
            )

            total_seconds = time.perf_counter() - total_started

        except (RuntimeError, OSError, MemoryError) as exc:
            raise CUDAExecutionError(
                f"Ошибка выполнения матричного умножения CUDA: {exc}"
            ) from exc

        if compute_seconds <= 0:
            compute_seconds = 1e-12

        transfer_seconds = h2d_seconds + d2h_seconds

        # Из-за особенностей таймера суммарные измерения могут отличаться
        # на доли микросекунды. Не допускаем нарушения контракта метрик.
        minimum_total = compute_seconds + transfer_seconds
        total_seconds = max(total_seconds, minimum_total)

        operations = self._matmul_operations(left, right)
        input_bytes = self._tensor_bytes(left) + self._tensor_bytes(right)
        output_bytes = self._tensor_bytes(result_cpu)

        return MatmulExecution(
            result=result_cpu,
            backend=ComputeBackend.CUDA,
            compute_seconds=compute_seconds,
            transfer_seconds=transfer_seconds,
            total_seconds=total_seconds,
            peak_memory_bytes=peak_memory,
            operations=operations,
            input_bytes=input_bytes,
            output_bytes=output_bytes,
        )

    @staticmethod
    def _matmul_operations(left: Any, right: Any) -> int:
        """Оценивает число арифметических операций матричного умножения."""
        rows = int(left.shape[0])
        inner = int(left.shape[1])
        columns = int(right.shape[1])
        return 2 * rows * inner * columns

    @staticmethod
    def _tensor_bytes(tensor: Any) -> int:
        """Возвращает размер тензора в байтах."""
        return int(tensor.numel() * tensor.element_size())

    def _load_torch(self) -> Any:
        """Лениво загружает PyTorch."""
        if self._torch_module is None:
            try:
                self._torch_module = importlib.import_module("torch")
            except ImportError as exc:
                raise BackendUnavailableError(
                    "PyTorch не установлен в текущем окружении."
                ) from exc

        return self._torch_module
