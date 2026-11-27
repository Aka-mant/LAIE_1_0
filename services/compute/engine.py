
"""Локальный вычислительный движок LAIE: CUDA с безопасным CPU fallback."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ComputeDeviceInfo:
    """Сведения о вычислительном устройстве."""

    backend: str
    available: bool
    device_name: str
    total_memory_mb: int | None = None
    free_memory_mb: int | None = None
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComputeDecision:
    """Результат выбора вычислительного устройства."""

    device: str
    reason: str
    required_memory_mb: int
    available_memory_mb: int | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ComputeEngine:
    """Выбирает CPU или CUDA с учётом размера задачи и памяти GPU."""

    VALID_FORCE_MODES = {"auto", "cpu", "cuda"}

    def __init__(
        self,
        min_gpu_items: int = 100_000,
        memory_reserve_mb: int = 1_536,
    ):
        if min_gpu_items < 1:
            raise ValueError("min_gpu_items должен быть положительным")
        if memory_reserve_mb < 0:
            raise ValueError("memory_reserve_mb не может быть отрицательным")

        self.min_gpu_items = min_gpu_items
        self.memory_reserve_mb = memory_reserve_mb
        self._torch = None
        self._torch_checked = False
        self.last_decision: ComputeDecision | None = None

    def _load_torch(self):
        """Ленивая загрузка PyTorch без CUDA-инициализации при импорте."""
        if not self._torch_checked:
            self._torch_checked = True
            try:
                import torch
            except (ImportError, OSError):
                self._torch = None
            else:
                self._torch = torch
        return self._torch

    def inspect(self) -> dict[str, Any]:
        """Получает состояние CUDA и доступной видеопамяти."""
        torch = self._load_torch()

        if torch is None:
            return {
                "selected": "cpu",
                "cuda_available": False,
                "torch_available": False,
                "gpu": ComputeDeviceInfo(
                    backend="cuda",
                    available=False,
                    device_name="Недоступно",
                    reason="PyTorch не установлен или его зависимости недоступны.",
                ).as_dict(),
                "cpu": ComputeDeviceInfo(
                    backend="cpu",
                    available=True,
                    device_name="CPU",
                    reason="Резервное вычислительное устройство.",
                ).as_dict(),
            }

        cuda_available = False
        gpu_info = ComputeDeviceInfo(
            backend="cuda",
            available=False,
            device_name="Недоступно",
            reason="CUDA не обнаружена.",
        )

        try:
            cuda_available = bool(torch.cuda.is_available())

            if cuda_available:
                index = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(index)
                free_bytes, total_bytes = torch.cuda.mem_get_info(index)

                gpu_info = ComputeDeviceInfo(
                    backend="cuda",
                    available=True,
                    device_name=props.name,
                    total_memory_mb=total_bytes // (1024 * 1024),
                    free_memory_mb=free_bytes // (1024 * 1024),
                    reason="CUDA доступна.",
                )
        except Exception as exc:
            cuda_available = False
            gpu_info = ComputeDeviceInfo(
                backend="cuda",
                available=False,
                device_name="Ошибка опроса GPU",
                reason=f"{type(exc).__name__}: {exc}",
            )

        return {
            "selected": "cuda" if cuda_available else "cpu",
            "cuda_available": cuda_available,
            "torch_available": True,
            "gpu": gpu_info.as_dict(),
            "cpu": ComputeDeviceInfo(
                backend="cpu",
                available=True,
                device_name="CPU",
                reason="Резервное вычислительное устройство.",
            ).as_dict(),
        }

    def explain_choice(
        self,
        item_count: int,
        estimated_memory_mb: int = 0,
        workspace_memory_mb: int = 0,
        force: str = "auto",
    ) -> ComputeDecision:
        """Объясняет выбор устройства и учитывает дополнительную память.

        estimated_memory_mb — базовая оценка памяти задачи.
        workspace_memory_mb — дополнительная память для промежуточных
        результатов и рабочих буферов.

        Принудительный CUDA не отключает проверку доступной VRAM.
        """
        if item_count < 0:
            raise ValueError("Размер задачи не может быть отрицательным")
        if estimated_memory_mb < 0 or workspace_memory_mb < 0:
            raise ValueError("Оценка памяти не может быть отрицательной")
        if force not in self.VALID_FORCE_MODES:
            raise ValueError("force должен быть auto, cpu или cuda")

        required_memory = estimated_memory_mb + workspace_memory_mb

        if force == "cpu":
            decision = ComputeDecision(
                "cpu",
                "CPU выбран принудительно.",
                required_memory,
                None,
            )
            self.last_decision = decision
            return decision

        state = self.inspect()

        if not state["cuda_available"]:
            decision = ComputeDecision(
                "cpu",
                "CUDA недоступна. Используется CPU.",
                required_memory,
                None,
            )
            self.last_decision = decision
            return decision

        gpu = state["gpu"]
        free_mb = gpu["free_memory_mb"]

        usable_memory = max(0, free_mb - self.memory_reserve_mb)

        if required_memory > usable_memory:
            decision = ComputeDecision(
                "cpu",
                (
                    "Недостаточно свободной VRAM с учётом резерва. "
                    f"Требуется {required_memory} МБ, "
                    f"доступно для задачи {usable_memory} МБ."
                ),
                required_memory,
                free_mb,
            )
            self.last_decision = decision
            return decision

        if force == "cuda":
            reason = "CUDA выбрана принудительно; проверка памяти пройдена."
        elif item_count < self.min_gpu_items:
            decision = ComputeDecision(
                "cpu",
                (
                    f"Задача меньше порога GPU: {item_count} "
                    f"< {self.min_gpu_items}. Используется CPU."
                ),
                required_memory,
                free_mb,
            )
            self.last_decision = decision
            return decision
        else:
            reason = "Размер задачи достаточен, запас VRAM позволяет использовать CUDA."

        decision = ComputeDecision(
            "cuda",
            reason,
            required_memory,
            free_mb,
        )
        self.last_decision = decision
        return decision

    def choose_device(
        self,
        item_count: int,
        estimated_memory_mb: int = 0,
        force: str = "auto",
        workspace_memory_mb: int = 0,
    ) -> str:
        """Возвращает cuda или cpu, сохраняя совместимость старого API."""
        return self.explain_choice(
            item_count=item_count,
            estimated_memory_mb=estimated_memory_mb,
            workspace_memory_mb=workspace_memory_mb,
            force=force,
        ).device

    def tensor_matmul(self, left, right, force: str = "auto"):
        """Выполняет матричное умножение с повтором на CPU при ошибке CUDA."""
        torch = self._load_torch()

        if torch is None:
            raise RuntimeError(
                "Для tensor_matmul необходимо установить PyTorch."
            )

        if left.ndim != 2 or right.ndim != 2:
            raise ValueError("Обе матрицы должны быть двумерными.")

        if left.shape[1] != right.shape[0]:
            raise ValueError("Несовместимые размеры матриц.")

        if left.device.type != "cpu" or right.device.type != "cpu":
            raise ValueError(
                "Входные матрицы должны находиться в CPU-памяти."
            )

        output_elements = left.shape[0] * right.shape[1]
        item_count = left.shape[0] * left.shape[1] * right.shape[1]

        input_bytes = (
            left.numel() * left.element_size()
            + right.numel() * right.element_size()
        )
        output_bytes = output_elements * max(
            left.element_size(), right.element_size()
        )

        # Учитываем входы, выход и дополнительный рабочий буфер.
        estimated_mb = (input_bytes + output_bytes) // (1024 * 1024)
        workspace_mb = (output_bytes * 2) // (1024 * 1024)

        decision = self.explain_choice(
            item_count=item_count,
            estimated_memory_mb=estimated_mb,
            workspace_memory_mb=workspace_mb,
            force=force,
        )

        if decision.device == "cpu":
            return left @ right

        try:
            result = left.to("cuda") @ right.to("cuda")
            torch.cuda.synchronize()
            return result

        except (RuntimeError, OSError):
            # CUDA могла исчерпать память или потерять доступность.
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

            self.last_decision = ComputeDecision(
                "cpu",
                "Ошибка CUDA во время операции. Выполнен переход на CPU.",
                decision.required_memory_mb,
                decision.available_memory_mb,
            )

            return left @ right
