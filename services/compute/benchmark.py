
"""Локальные бенчмарки CPU/GPU для LAIE.

Модуль измеряет производительность матричного умножения
на CPU и CUDA, проверяет численную точность относительно
эталонного вычисления float64 и сохраняет метрики ошибок.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any

from services.compute.engine import ComputeEngine


@dataclass(frozen=True)
class BenchmarkResult:
    """Результаты измерения одного вычислительного устройства."""

    device: str
    size: int
    repeats: int
    warmup: int
    median_seconds: float
    min_seconds: float
    max_seconds: float

    max_absolute_error: float | None = None
    correct: bool | None = None

    rmse: float | None = None
    relative_l2_error: float | None = None
    percentile_999_absolute_error: float | None = None

    free_memory_mb_before: int | None = None
    free_memory_mb_after: int | None = None

    def as_dict(self) -> dict[str, Any]:
        """Возвращает сериализуемое представление результата."""
        return asdict(self)


class BenchmarkError(RuntimeError):
    """Ошибка выполнения бенчмарка."""


class BenchmarkRunner:
    """Измеряет производительность локальных CPU и CUDA."""

    # Максимально допустимая относительная L2-ошибка
    # для данного теста матричного умножения float32.
    MAX_RELATIVE_L2_ERROR = 1e-5

    def __init__(
        self,
        engine: ComputeEngine | None = None,
    ):
        self.engine = engine or ComputeEngine()

    @staticmethod
    def _validate(
        size: int,
        repeats: int,
        warmup: int,
    ) -> None:
        """Проверяет параметры бенчмарка."""
        if size < 2:
            raise ValueError(
                "Размер матрицы должен быть не меньше 2."
            )

        if repeats < 1:
            raise ValueError(
                "Количество повторов должно быть не меньше 1."
            )

        if warmup < 0:
            raise ValueError(
                "Количество прогревочных запусков "
                "не может быть отрицательным."
            )

    @staticmethod
    def _memory_snapshot(
        torch,
        device: str,
    ) -> int | None:
        """Возвращает количество свободной VRAM в мегабайтах."""
        if device != "cuda":
            return None

        free_bytes, _ = torch.cuda.mem_get_info()

        return int(
            free_bytes // (1024 * 1024)
        )

    @staticmethod
    def _measure_accuracy(
        torch,
        actual,
        reference,
    ) -> dict[str, Any]:
        """Оценивает точность относительно эталона float64.

        Возвращает:
        - максимальную абсолютную ошибку;
        - RMSE;
        - относительную L2-ошибку;
        - 99,9-й процентиль абсолютной ошибки;
        - признак прохождения проверки.

        Проверка не зависит от масштаба отдельного элемента:
        основной критерий использует относительную L2-ошибку.
        """
        actual = actual.to(
            device="cpu",
            dtype=torch.float64,
        )

        reference = reference.to(
            device="cpu",
            dtype=torch.float64,
        )

        if not bool(torch.isfinite(actual).all().item()):
            return {
                "max_absolute_error": float("inf"),
                "rmse": float("inf"),
                "relative_l2_error": float("inf"),
                "percentile_999_absolute_error": float("inf"),
                "correct": False,
            }

        difference = actual - reference
        absolute_difference = difference.abs()

        max_error = float(
            absolute_difference.max().item()
        )

        rmse = float(
            torch.sqrt(
                torch.mean(difference ** 2)
            ).item()
        )

        reference_norm = torch.linalg.vector_norm(
            reference
        )

        difference_norm = torch.linalg.vector_norm(
            difference
        )

        if reference_norm.item() == 0:
            relative_l2_error = (
                0.0
                if difference_norm.item() == 0
                else float("inf")
            )
        else:
            relative_l2_error = float(
                (
                    difference_norm / reference_norm
                ).item()
            )

        percentile_999 = float(
            torch.quantile(
                absolute_difference.flatten(),
                0.999,
            ).item()
        )

        correct = bool(
            relative_l2_error
            <= BenchmarkRunner.MAX_RELATIVE_L2_ERROR
        )

        return {
            "max_absolute_error": max_error,
            "rmse": rmse,
            "relative_l2_error": relative_l2_error,
            "percentile_999_absolute_error": percentile_999,
            "correct": correct,
        }

    def _run_device(
        self,
        torch,
        size: int,
        repeats: int,
        warmup: int,
        device: str,
    ) -> tuple[BenchmarkResult, Any]:
        """Выполняет измерения производительности на устройстве."""

        free_before = self._memory_snapshot(
            torch,
            device,
        )

        # Генерация одинаковых исходных данных для CPU и CUDA.
        generator = torch.Generator(
            device="cpu"
        )

        generator.manual_seed(42)

        left_cpu = torch.randn(
            (size, size),
            generator=generator,
            dtype=torch.float32,
        )

        right_cpu = torch.randn(
            (size, size),
            generator=generator,
            dtype=torch.float32,
        )

        # Перенос входных данных на выбранное устройство.
        # Время переноса не включается в измерения.
        left = left_cpu.to(device)
        right = right_cpu.to(device)

        if device == "cuda":
            torch.cuda.synchronize()

        # Прогревочные запуски.
        for _ in range(warmup):
            result = left @ right

        if device == "cuda":
            torch.cuda.synchronize()

        durations: list[float] = []

        # Основные измерения.
        for _ in range(repeats):
            if device == "cuda":
                torch.cuda.synchronize()

            started = time.perf_counter()

            result = left @ right

            if device == "cuda":
                torch.cuda.synchronize()

            durations.append(
                time.perf_counter() - started
            )

        # Эталонное вычисление float64.
        # Выполняется вне измеряемого интервала.
        reference = (
            left_cpu.to(torch.float64)
            @ right_cpu.to(torch.float64)
        )

        actual = result.detach().to(
            device="cpu",
            dtype=torch.float64,
        )

        # Расчёт метрик точности.
        accuracy = self._measure_accuracy(
            torch=torch,
            actual=actual,
            reference=reference,
        )

        free_after = self._memory_snapshot(
            torch,
            device,
        )

        benchmark_result = BenchmarkResult(
            device=device,
            size=size,
            repeats=repeats,
            warmup=warmup,
            median_seconds=statistics.median(durations),
            min_seconds=min(durations),
            max_seconds=max(durations),
            max_absolute_error=accuracy[
                "max_absolute_error"
            ],
            correct=accuracy["correct"],
            rmse=accuracy["rmse"],
            relative_l2_error=accuracy[
                "relative_l2_error"
            ],
            percentile_999_absolute_error=accuracy[
                "percentile_999_absolute_error"
            ],
            free_memory_mb_before=free_before,
            free_memory_mb_after=free_after,
        )

        return benchmark_result, result

    def run_matmul(
        self,
        size: int = 2048,
        repeats: int = 5,
        warmup: int = 2,
        devices: tuple[str, ...] = ("cpu", "cuda"),
    ) -> dict[str, Any]:
        """Сравнивает матричное умножение CPU и CUDA.

        Передача входных данных на GPU не включается
        в измерение времени вычисления.

        CPU запускается первым и используется как
        отдельное контрольное устройство.

        Эталон точности рассчитывается в float64.
        """

        self._validate(
            size=size,
            repeats=repeats,
            warmup=warmup,
        )

        if not devices or any(
            device not in {"cpu", "cuda"}
            for device in devices
        ):
            raise ValueError(
                "devices должны содержать cpu и/или cuda."
            )

        if len(set(devices)) != len(devices):
            raise ValueError(
                "Устройства не должны повторяться."
            )

        try:
            import torch
        except (ImportError, OSError) as exc:
            raise BenchmarkError(
                "PyTorch недоступен."
            ) from exc

        if not hasattr(torch, "cuda"):
            raise BenchmarkError(
                "Установленная сборка PyTorch "
                "не поддерживает CUDA API."
            )

        results: dict[str, dict[str, Any]] = {}
        skipped: dict[str, str] = {}

        cpu_result = None

        # CPU запускается первым.
        ordered_devices = sorted(
            devices,
            key=lambda item: item != "cpu",
        )

        for device in ordered_devices:

            if device == "cuda":
                state = self.engine.inspect()

                if not state.get("cuda_available"):
                    skipped["cuda"] = "CUDA недоступна."
                    continue

                # Оценка памяти:
                # две входные матрицы,
                # результирующая матрица,
                # дополнительный рабочий буфер.
                matrix_bytes = size * size * 4

                estimated_mb = self.engine.bytes_to_mb_ceil(
                    matrix_bytes * 4
                )

                decision = self.engine.explain_choice(
                    item_count=size * size * size,
                    estimated_memory_mb=estimated_mb,
                    workspace_memory_mb=0,
                    force="cuda",
                )

                if decision.device != "cuda":
                    skipped["cuda"] = decision.reason
                    continue

            try:
                result, tensor = self._run_device(
                    torch=torch,
                    size=size,
                    repeats=repeats,
                    warmup=warmup,
                    device=device,
                )

            except (RuntimeError, OSError) as exc:

                if device == "cuda":
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

                    skipped["cuda"] = (
                        f"{type(exc).__name__}: {exc}"
                    )

                    continue

                raise BenchmarkError(
                    "Ошибка CPU-бенчмарка: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

            results[device] = result.as_dict()

            if device == "cpu":
                cpu_result = result

            del tensor

            if device == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        # Отношение медианного времени CPU к CUDA.
        speedup = None

        if (
            cpu_result is not None
            and "cuda" in results
            and results["cuda"]["median_seconds"] > 0
        ):
            speedup = (
                cpu_result.median_seconds
                / results["cuda"]["median_seconds"]
            )

        return {
            "benchmark": "matmul_float32",
            "size": size,
            "repeats": repeats,
            "warmup": warmup,
            "accuracy_reference": "float64",
            "max_relative_l2_error": (
                self.MAX_RELATIVE_L2_ERROR
            ),
            "results": results,
            "skipped": skipped,
            "gpu_speedup": speedup,
            "speedup_note": (
                "Отношение медианного времени CPU к "
                "медианному времени CUDA. Передача "
                "данных между CPU и GPU не включена."
            ),
        }
