
"""Консервативная оценка потребления памяти вычислительными задачами."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from .task_types import ComputeBackend, TaskType


MIB = 1024 * 1024


@dataclass(frozen=True, slots=True)
class MemoryDeviceSnapshot:
    """Снимок общей и свободной памяти вычислительного устройства."""

    backend: ComputeBackend
    total_bytes: int
    free_bytes: int
    device_name: str = "unknown"

    def __post_init__(self) -> None:
        if not isinstance(self.backend, ComputeBackend):
            raise TypeError("backend должен быть ComputeBackend.")

        for field_name in ("total_bytes", "free_bytes"):
            value = getattr(self, field_name)

            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} должен быть целым числом.")

            if value < 0:
                raise ValueError(f"{field_name} не может быть отрицательным.")

        if self.total_bytes <= 0:
            raise ValueError("total_bytes должен быть больше нуля.")

        if self.free_bytes > self.total_bytes:
            raise ValueError("free_bytes не может превышать total_bytes.")

        if not isinstance(self.device_name, str) or not self.device_name.strip():
            raise ValueError("device_name должен быть непустой строкой.")


@dataclass(frozen=True, slots=True)
class MemoryEstimate:
    """Результат оценки памяти для конкретной задачи и устройства."""

    task_type: TaskType
    backend: ComputeBackend
    device_name: str

    estimated_payload_bytes: int
    runtime_overhead_bytes: int
    safety_margin_bytes: int
    required_bytes: int

    total_device_bytes: int
    free_device_bytes: int
    reserve_bytes: int
    usable_bytes: int

    fits: bool
    recommended_batch_size: int | None
    batch_required_bytes: int | None

    confidence: str = "heuristic"

    @property
    def required_mib(self) -> float:
        """Расчётная потребность в MiB."""
        return self.required_bytes / MIB

    @property
    def usable_mib(self) -> float:
        """Доступный объём памяти после вычета резерва."""
        return self.usable_bytes / MIB

    @property
    def free_after_reserve_bytes(self) -> int:
        """Свободная память после вычета резерва."""
        return self.usable_bytes


    def as_dict(self) -> dict[str, Any]:
        """Возвращает JSON-совместимое представление результата."""
        result = asdict(self)
        result["task_type"] = self.task_type.value
        result["backend"] = self.backend.value
        result["required_mib"] = round(self.required_mib, 3)
        result["usable_mib"] = round(self.usable_mib, 3)
        return result



class MemoryEstimator:
    """Консервативно оценивает память CPU/CUDA и возможность пакетирования."""

    def __init__(
        self,
        *,
        safety_margin_ratio: float = 0.20,
        cpu_reserve_bytes: int = 512 * MIB,
        cuda_reserve_bytes: int = 1536 * MIB,
        cpu_runtime_overhead_bytes: int = 64 * MIB,
        cuda_runtime_overhead_bytes: int = 256 * MIB,
    ) -> None:
        self._validate_ratio(safety_margin_ratio)

        self._validate_non_negative_int(
            "cpu_reserve_bytes",
            cpu_reserve_bytes,
        )
        self._validate_non_negative_int(
            "cuda_reserve_bytes",
            cuda_reserve_bytes,
        )
        self._validate_non_negative_int(
            "cpu_runtime_overhead_bytes",
            cpu_runtime_overhead_bytes,
        )
        self._validate_non_negative_int(
            "cuda_runtime_overhead_bytes",
            cuda_runtime_overhead_bytes,
        )

        self.safety_margin_ratio = safety_margin_ratio
        self.cpu_reserve_bytes = cpu_reserve_bytes
        self.cuda_reserve_bytes = cuda_reserve_bytes
        self.cpu_runtime_overhead_bytes = cpu_runtime_overhead_bytes
        self.cuda_runtime_overhead_bytes = cuda_runtime_overhead_bytes

    @staticmethod
    def _validate_non_negative_int(name: str, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} должен быть целым числом.")

        if value < 0:
            raise ValueError(f"{name} не может быть отрицательным.")

    @staticmethod
    def _validate_ratio(value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("safety_margin_ratio должен быть числом.")

        if not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError(
                "safety_margin_ratio должен находиться в диапазоне [0, 1)."
            )

    def _get_reserve(self, backend: ComputeBackend) -> int:
        if backend == ComputeBackend.CUDA:
            return self.cuda_reserve_bytes

        if backend == ComputeBackend.CPU:
            return self.cpu_reserve_bytes

        raise ValueError(f"Неподдерживаемый backend: {backend!r}")

    def _get_runtime_overhead(self, backend: ComputeBackend) -> int:
        if backend == ComputeBackend.CUDA:
            return self.cuda_runtime_overhead_bytes

        if backend == ComputeBackend.CPU:
            return self.cpu_runtime_overhead_bytes

        raise ValueError(f"Неподдерживаемый backend: {backend!r}")

    @staticmethod
    def _get_payload_bytes(analysis: Any) -> int:
        """Берёт наиболее консервативную оценку объёма рабочих данных."""
        components = (
            analysis.estimated_input_bytes
            + analysis.estimated_output_bytes
            + analysis.estimated_workspace_bytes
        )

        return max(analysis.estimated_memory_bytes, components)

    def estimate(
        self,
        analysis: Any,
        snapshot: MemoryDeviceSnapshot,
    ) -> MemoryEstimate:
        """Оценивает полную потребность задачи в памяти на устройстве."""
        if not isinstance(snapshot, MemoryDeviceSnapshot):
            raise TypeError("snapshot должен быть MemoryDeviceSnapshot.")

        if snapshot.backend not in analysis.supported_backends:
            raise ValueError(
                f"Задача {analysis.task_type.value} "
                f"не поддерживает {snapshot.backend.value}."
            )

        payload_bytes = self._get_payload_bytes(analysis)
        runtime_overhead = self._get_runtime_overhead(snapshot.backend)

        safety_margin = math.ceil(
            payload_bytes * self.safety_margin_ratio
        )

        required_bytes = (
            payload_bytes
            + safety_margin
            + runtime_overhead
        )

        reserve_bytes = self._get_reserve(snapshot.backend)
        usable_bytes = max(
            0,
            snapshot.free_bytes - reserve_bytes,
        )

        fits = required_bytes <= usable_bytes

        batch_size: int | None = None
        batch_required_bytes: int | None = None

        if analysis.supports_batching:
            batch_size, batch_required_bytes = self._recommend_batch_size(
                analysis=analysis,
                payload_bytes=payload_bytes,
                runtime_overhead_bytes=runtime_overhead,
                usable_bytes=usable_bytes,
            )

        return MemoryEstimate(
            task_type=analysis.task_type,
            backend=snapshot.backend,
            device_name=snapshot.device_name,
            estimated_payload_bytes=payload_bytes,
            runtime_overhead_bytes=runtime_overhead,
            safety_margin_bytes=safety_margin,
            required_bytes=required_bytes,
            total_device_bytes=snapshot.total_bytes,
            free_device_bytes=snapshot.free_bytes,
            reserve_bytes=reserve_bytes,
            usable_bytes=usable_bytes,
            fits=fits,
            recommended_batch_size=batch_size,
            batch_required_bytes=batch_required_bytes,
        )

    def _recommend_batch_size(
        self,
        *,
        analysis: Any,
        payload_bytes: int,
        runtime_overhead_bytes: int,
        usable_bytes: int,
    ) -> tuple[int | None, int | None]:
        """Рассчитывает консервативный размер пакета по линейной модели."""
        item_count = analysis.item_count

        if item_count <= 0:
            return None, None

        if usable_bytes <= runtime_overhead_bytes:
            return None, None

        bytes_per_item = payload_bytes / item_count

        if bytes_per_item <= 0:
            return item_count, runtime_overhead_bytes

        available_for_payload = usable_bytes - runtime_overhead_bytes

        safe_bytes_per_item = (
            bytes_per_item * (1 + self.safety_margin_ratio)
        )

        batch_size = math.floor(
            available_for_payload / safe_bytes_per_item
        )

        if batch_size < 1:
            return None, None

        batch_size = min(batch_size, item_count)

        batch_payload = math.ceil(
            payload_bytes * batch_size / item_count
        )
        batch_margin = math.ceil(
            batch_payload * self.safety_margin_ratio
        )

        batch_required = (
            batch_payload
            + batch_margin
            + runtime_overhead_bytes
        )

        # Защита от погрешности округления.
        while batch_size > 0 and batch_required > usable_bytes:
            batch_size -= 1

            if batch_size == 0:
                return None, None

            batch_payload = math.ceil(
                payload_bytes * batch_size / item_count
            )
            batch_margin = math.ceil(
                batch_payload * self.safety_margin_ratio
            )
            batch_required = (
                batch_payload
                + batch_margin
                + runtime_overhead_bytes
            )

        return batch_size, batch_required
