
"""Локальный мониторинг вычислительного оборудования LAIE.

Не требует psutil. В Windows получает сведения об оперативной памяти
через GlobalMemoryStatusEx, сведения о CUDA — через ленивую загрузку PyTorch.
"""

from __future__ import annotations

import ctypes
import importlib
import os
import platform
from dataclasses import dataclass
from typing import Any

from .memory_estimator import MemoryDeviceSnapshot
from .task_types import ComputeBackend


@dataclass(frozen=True, slots=True)
class HardwareSnapshot:
    """Снимок характеристик локального компьютера."""

    cpu_name: str
    cpu_logical_cores: int
    ram_total_bytes: int
    ram_available_bytes: int
    cuda_available: bool
    cuda_device_name: str | None = None
    cuda_total_bytes: int | None = None
    cuda_free_bytes: int | None = None
    cuda_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.cpu_name.strip():
            raise ValueError("cpu_name не может быть пустым.")

        if self.cpu_logical_cores <= 0:
            raise ValueError("cpu_logical_cores должен быть больше нуля.")

        if self.ram_total_bytes <= 0:
            raise ValueError("ram_total_bytes должен быть больше нуля.")

        if not 0 <= self.ram_available_bytes <= self.ram_total_bytes:
            raise ValueError("Некорректное значение доступной RAM.")

        if self.cuda_available:
            if not self.cuda_device_name:
                raise ValueError("Для CUDA необходимо имя устройства.")
            if self.cuda_total_bytes is None or self.cuda_total_bytes <= 0:
                raise ValueError("Для CUDA необходим объём VRAM.")
            if self.cuda_free_bytes is None:
                raise ValueError("Для CUDA необходим объём свободной VRAM.")
            if not 0 <= self.cuda_free_bytes <= self.cuda_total_bytes:
                raise ValueError("Некорректное значение свободной VRAM.")

    def memory_snapshots(self) -> tuple[MemoryDeviceSnapshot, ...]:
        """Преобразует сведения о памяти в формат MemoryEstimator."""
        snapshots = [
            MemoryDeviceSnapshot(
                backend=ComputeBackend.CPU,
                total_bytes=self.ram_total_bytes,
                free_bytes=self.ram_available_bytes,
                device_name=self.cpu_name,
            )
        ]

        if self.cuda_available:
            snapshots.append(
                MemoryDeviceSnapshot(
                    backend=ComputeBackend.CUDA,
                    total_bytes=self.cuda_total_bytes,
                    free_bytes=self.cuda_free_bytes,
                    device_name=self.cuda_device_name,
                )
            )

        return tuple(snapshots)

    def as_dict(self) -> dict[str, Any]:
        """Возвращает JSON-совместимое представление."""
        return {
            "cpu": {
                "name": self.cpu_name,
                "logical_cores": self.cpu_logical_cores,
            },
            "ram": {
                "total_bytes": self.ram_total_bytes,
                "available_bytes": self.ram_available_bytes,
            },
            "cuda": {
                "available": self.cuda_available,
                "device_name": self.cuda_device_name,
                "total_bytes": self.cuda_total_bytes,
                "free_bytes": self.cuda_free_bytes,
                "reason": self.cuda_reason,
            },
        }


class HardwareMonitor:
    """Получает актуальный снимок локального оборудования."""

    def __init__(self, torch_module: Any | None = None) -> None:
        self._torch_module = torch_module

    def inspect(self) -> HardwareSnapshot:
        """Собирает сведения о CPU, RAM и CUDA."""
        ram_total, ram_available = self._get_system_memory()

        cpu_name = (
            platform.processor().strip()
            or platform.machine().strip()
            or "Unknown CPU"
        )
        cpu_cores = os.cpu_count() or 1

        cuda_data = self._inspect_cuda()

        return HardwareSnapshot(
            cpu_name=cpu_name,
            cpu_logical_cores=cpu_cores,
            ram_total_bytes=ram_total,
            ram_available_bytes=ram_available,
            cuda_available=cuda_data["available"],
            cuda_device_name=cuda_data["device_name"],
            cuda_total_bytes=cuda_data["total_bytes"],
            cuda_free_bytes=cuda_data["free_bytes"],
            cuda_reason=cuda_data["reason"],
        )

    def get_memory_snapshots(
        self,
    ) -> tuple[MemoryDeviceSnapshot, ...]:
        """Возвращает снимки памяти для MemoryEstimator."""
        return self.inspect().memory_snapshots()

    @staticmethod
    def _get_system_memory() -> tuple[int, int]:
        """Возвращает (всего, доступно) в байтах."""
        if os.name == "nt":
            return HardwareMonitor._get_windows_memory()

        return HardwareMonitor._get_posix_memory()

    @staticmethod
    def _get_windows_memory() -> tuple[int, int]:
        """Получает память Windows через GlobalMemoryStatusEx."""

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        function = kernel32.GlobalMemoryStatusEx
        function.argtypes = [ctypes.POINTER(MEMORYSTATUSEX)]
        function.restype = ctypes.c_int

        if not function(ctypes.byref(status)):
            error_code = ctypes.get_last_error()
            raise OSError(
                error_code,
                "Не удалось получить сведения об оперативной памяти Windows.",
            )

        total = int(status.ullTotalPhys)
        available = int(status.ullAvailPhys)

        if total <= 0 or available < 0 or available > total:
            raise RuntimeError(
                "Windows вернула некорректные сведения об оперативной памяти."
            )

        return total, available

    @staticmethod
    def _get_posix_memory() -> tuple[int, int]:
        """Получает память через sysconf на POSIX-системах."""
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            total_pages = int(os.sysconf("SC_PHYS_PAGES"))
            available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        except (AttributeError, OSError, ValueError) as exc:
            raise RuntimeError(
                "Не удалось получить сведения об оперативной памяти."
            ) from exc

        total = page_size * total_pages
        available = page_size * available_pages

        if total <= 0 or not 0 <= available <= total:
            raise RuntimeError(
                "Система вернула некорректные сведения об оперативной памяти."
            )

        return total, available

    def _inspect_cuda(self) -> dict[str, Any]:
        """Проверяет CUDA и получает текущую свободную VRAM."""
        result: dict[str, Any] = {
            "available": False,
            "device_name": None,
            "total_bytes": None,
            "free_bytes": None,
            "reason": None,
        }

        try:
            torch = self._load_torch()

            if not torch.cuda.is_available():
                result["reason"] = (
                    "PyTorch не обнаружил доступное CUDA-устройство."
                )
                return result

            device_index = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(device_index)
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)

            result.update(
                {
                    "available": True,
                    "device_name": str(properties.name),
                    "total_bytes": int(total_bytes),
                    "free_bytes": int(free_bytes),
                    "reason": None,
                }
            )
            return result

        except ImportError:
            result["reason"] = "PyTorch не установлен."
        except (OSError, RuntimeError, AttributeError, ValueError) as exc:
            result["reason"] = (
                f"Не удалось получить сведения CUDA: {exc}"
            )

        return result

    def _load_torch(self) -> Any:
        """Загружает PyTorch только при первом обращении."""
        if self._torch_module is None:
            self._torch_module = importlib.import_module("torch")
        return self._torch_module
