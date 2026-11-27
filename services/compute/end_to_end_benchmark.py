
"""Сквозные локальные бенчмарки CPU/GPU для LAIE.

Измеряет полный вычислительный маршрут:
CPU RAM -> GPU VRAM -> CUDA вычисления -> CPU RAM.

Все операции выполняются локально.
Сетевые и облачные сервисы не используются.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any

from services.compute.engine import ComputeEngine


@dataclass(frozen=True)
class EndToEndBenchmarkResult:
    """Результаты сквозного измерения одного устройства."""

    device: str
    size: int
    repeats: int
    warmup: int

    median_total_seconds: float
    min_total_seconds: float
    max_total_seconds: float

    median_compute_seconds: float
    median_input_transfer_seconds: float
    median_output_transfer_seconds: float

    max_absolute_error: float
    rmse: float
    relative_l2_error: float
    percentile_999_absolute_error: float
    correct: bool

    free_memory_mb_before: int | None = None
    free_memory_mb_after: int | None = None

    def as_dict(self) -> dict[str, Any]:
        """Возвращает сериализуемое представление результата."""
        return asdict(self)


class EndToEndBenchmarkError(RuntimeError):
    """Ошибка выполнения сквозного бенчмарка."""


class EndToEndBenchmarkRunner:
    """Сравнивает полное время выполнения CPU и CUDA."""

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
        """Проверяет параметры измерения."""
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
        """Возвращает свободную видеопамять в мегабайтах."""
        if device != "cuda":
            return None

        free_bytes, _ = torch.cuda.mem_get_info()

        return int(
            free_bytes // (1024 * 1024)
        )

    @classmethod
    def _measure_accuracy(
        cls,
        torch,
        actual,
        reference,
    ) -> dict[str, Any]:
        """Сравнивает результат с эталоном float64."""
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
            <= cls.MAX_RELATIVE_L2_ERROR
        )

        return {
            "max_absolute_error": max_error,
            "rmse": rmse,
            "relative_l2_error": relative_l2_error,
            "percentile_999_absolute_error": percentile_999,
            "correct": correct,
        }

    @staticmethod
    def _build_result(
        device: str,
        size: int,
        repeats: int,
        warmup: int,
        total_times: list[float],
        compute_times: list[float],
        input_transfer_times: list[float],
        output_transfer_times: list[float],
        accuracy: dict[str, Any],
        free_before: int | None,
        free_after: int | None,
    ) -> EndToEndBenchmarkResult:
        """Формирует итоговый объект результатов."""
        return EndToEndBenchmarkResult(
            device=device,
            size=size,
            repeats=repeats,
            warmup=warmup,
            median_total_seconds=statistics.median(
                total_times
            ),
            min_total_seconds=min(total_times),
            max_total_seconds=max(total_times),
            median_compute_seconds=statistics.median(
                compute_times
            ),
            median_input_transfer_seconds=statistics.median(
                input_transfer_times
            ),
            median_output_transfer_seconds=statistics.median(
                output_transfer_times
            ),
            max_absolute_error=accuracy[
                "max_absolute_error"
            ],
            rmse=accuracy["rmse"],
            relative_l2_error=accuracy[
                "relative_l2_error"
            ],
            percentile_999_absolute_error=accuracy[
                "percentile_999_absolute_error"
            ],
            correct=accuracy["correct"],
            free_memory_mb_before=free_before,
            free_memory_mb_after=free_after,
        )

    def _run_cpu(
        self,
        torch,
        left_cpu,
        right_cpu,
        reference,
        size: int,
        repeats: int,
        warmup: int,
    ) -> tuple[EndToEndBenchmarkResult, Any]:
        """Измеряет CPU-маршрут."""
        total_times: list[float] = []
        compute_times: list[float] = []

        # Прогрев CPU.
        for _ in range(warmup):
            result = left_cpu @ right_cpu

        # Основные измерения.
        for _ in range(repeats):
            started = time.perf_counter()

            result = left_cpu @ right_cpu

            elapsed = time.perf_counter() - started

            total_times.append(elapsed)
            compute_times.append(elapsed)

        accuracy = self._measure_accuracy(
            torch=torch,
            actual=result,
            reference=reference,
        )

        zero_times = [0.0] * repeats

        benchmark_result = self._build_result(
            device="cpu",
            size=size,
            repeats=repeats,
            warmup=warmup,
            total_times=total_times,
            compute_times=compute_times,
            input_transfer_times=zero_times,
            output_transfer_times=zero_times,
            accuracy=accuracy,
            free_before=None,
            free_after=None,
        )

        return benchmark_result, result

    def _run_cuda(
        self,
        torch,
        left_cpu,
        right_cpu,
        reference,
        size: int,
        repeats: int,
        warmup: int,
    ) -> tuple[EndToEndBenchmarkResult, Any]:
        """Измеряет полный маршрут CPU -> GPU -> CPU."""
        free_before = self._memory_snapshot(
            torch,
            "cuda",
        )

        total_times: list[float] = []
        compute_times: list[float] = []
        input_transfer_times: list[float] = []
        output_transfer_times: list[float] = []

        # Прогревочные запуски.
        # Передача данных включена в каждый прогревочный цикл.
        for _ in range(warmup):
            left_gpu = left_cpu.to("cuda")
            right_gpu = right_cpu.to("cuda")

            torch.cuda.synchronize()

            result_gpu = left_gpu @ right_gpu

            torch.cuda.synchronize()

            result_cpu = result_gpu.to("cpu")

            torch.cuda.synchronize()

            del left_gpu
            del right_gpu
            del result_gpu
            del result_cpu

        # Основные измерения.
        for _ in range(repeats):
            total_started = time.perf_counter()

            # CPU RAM -> GPU VRAM.
            transfer_started = time.perf_counter()

            left_gpu = left_cpu.to("cuda")
            right_gpu = right_cpu.to("cuda")

            torch.cuda.synchronize()

            input_transfer_elapsed = (
                time.perf_counter() - transfer_started
            )

            # Вычисление на GPU.
            compute_started = time.perf_counter()

            result_gpu = left_gpu @ right_gpu

            torch.cuda.synchronize()

            compute_elapsed = (
                time.perf_counter() - compute_started
            )

            # GPU VRAM -> CPU RAM.
            output_transfer_started = time.perf_counter()

            result_cpu = result_gpu.to("cpu")

            torch.cuda.synchronize()

            output_transfer_elapsed = (
                time.perf_counter() - output_transfer_started
            )

            total_elapsed = (
                time.perf_counter() - total_started
            )

            total_times.append(total_elapsed)
            compute_times.append(compute_elapsed)
            input_transfer_times.append(
                input_transfer_elapsed
            )
            output_transfer_times.append(
                output_transfer_elapsed
            )

            # Сохраняем результат последнего повтора.
            actual = result_cpu

            del left_gpu
            del right_gpu
            del result_gpu

        accuracy = self._measure_accuracy(
            torch=torch,
            actual=actual,
            reference=reference,
        )

        free_after = self._memory_snapshot(
            torch,
            "cuda",
        )

        benchmark_result = self._build_result(
            device="cuda",
            size=size,
            repeats=repeats,
            warmup=warmup,
            total_times=total_times,
            compute_times=compute_times,
            input_transfer_times=input_transfer_times,
            output_transfer_times=output_transfer_times,
            accuracy=accuracy,
            free_before=free_before,
            free_after=free_after,
        )

        return benchmark_result, actual

    def run_matmul(
        self,
        size: int = 2048,
        repeats: int = 5,
        warmup: int = 2,
        devices: tuple[str, ...] = ("cpu", "cuda"),
    ) -> dict[str, Any]:
        """Запускает сквозное сравнение CPU и CUDA.

        Генерация исходных данных и эталон float64
        выполняются вне измеряемого интервала.

        CPU:
            матричное умножение с выделением результата.

        CUDA:
            передача двух входных матриц;
            вычисление;
            синхронизация;
            возврат результата в CPU RAM.

        Полное время CUDA включает передачу данных
        и вычисление, а не только CUDA kernel.
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
            raise EndToEndBenchmarkError(
                "PyTorch недоступен."
            ) from exc

        if not hasattr(torch, "cuda"):
            raise EndToEndBenchmarkError(
                "PyTorch не поддерживает CUDA API."
            )

        # Создание исходных данных.
        generation_started = time.perf_counter()

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

        generation_seconds = (
            time.perf_counter() - generation_started
        )

        # Эталон float64 рассчитывается вне замеров.
        reference = (
            left_cpu.to(torch.float64)
            @ right_cpu.to(torch.float64)
        )

        results: dict[str, dict[str, Any]] = {}
        skipped: dict[str, str] = {}

        cpu_result = None

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

                # Оценка VRAM:
                # две входные матрицы,
                # выходная матрица,
                # дополнительный рабочий буфер.
                matrix_bytes = size * size * 4

                estimated_memory_mb = (
                    matrix_bytes * 4
                ) // (1024 * 1024)

                decision = self.engine.explain_choice(
                    item_count=size * size * size,
                    estimated_memory_mb=estimated_memory_mb,
                    workspace_memory_mb=0,
                    force="cuda",
                )

                if decision.device != "cuda":
                    skipped["cuda"] = decision.reason
                    continue

            try:
                if device == "cpu":
                    result, tensor = self._run_cpu(
                        torch=torch,
                        left_cpu=left_cpu,
                        right_cpu=right_cpu,
                        reference=reference,
                        size=size,
                        repeats=repeats,
                        warmup=warmup,
                    )

                    cpu_result = result

                else:
                    result, tensor = self._run_cuda(
                        torch=torch,
                        left_cpu=left_cpu,
                        right_cpu=right_cpu,
                        reference=reference,
                        size=size,
                        repeats=repeats,
                        warmup=warmup,
                    )

                results[device] = result.as_dict()

                del tensor

                if device == "cuda":
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

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

                raise EndToEndBenchmarkError(
                    "Ошибка CPU-бенчмарка: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

        # Сравниваем полное время, включая передачу данных.
        gpu_end_to_end_speedup = None

        if (
            cpu_result is not None
            and "cuda" in results
            and results["cuda"]["median_total_seconds"] > 0
        ):
            gpu_end_to_end_speedup = (
                cpu_result.median_total_seconds
                / results["cuda"]["median_total_seconds"]
            )

        return {
            "benchmark": "end_to_end_matmul_float32",
            "size": size,
            "repeats": repeats,
            "warmup": warmup,
            "input_generation_seconds": generation_seconds,
            "accuracy_reference": "float64",
            "max_relative_l2_error": (
                self.MAX_RELATIVE_L2_ERROR
            ),
            "results": results,
            "skipped": skipped,
            "gpu_end_to_end_speedup": (
                gpu_end_to_end_speedup
            ),
            "speedup_note": (
                "CPU/GPU сравниваются по полному времени. "
                "Для CUDA учитываются передача входных "
                "данных, вычисление, синхронизация "
                "и возврат результата в CPU RAM."
            ),
        }
