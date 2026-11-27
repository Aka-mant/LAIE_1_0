
"""Запуск сквозных бенчмарков LAIE через Django."""

import json
from datetime import datetime
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from services.compute.end_to_end_benchmark import (
    EndToEndBenchmarkError,
    EndToEndBenchmarkRunner,
)


class Command(BaseCommand):
    help = "Запускает сквозное сравнение CPU и CUDA."

    def add_arguments(self, parser):
        parser.add_argument(
            "--sizes",
            nargs="+",
            type=int,
            default=[256, 512, 1024, 2048],
            help="Размеры квадратных матриц.",
        )

        parser.add_argument(
            "--repeats",
            type=int,
            default=5,
            help="Количество измерений.",
        )

        parser.add_argument(
            "--warmup",
            type=int,
            default=2,
            help="Количество прогревочных запусков.",
        )

        parser.add_argument(
            "--device",
            choices=("both", "cpu", "cuda"),
            default="both",
            help="Устройство для измерений.",
        )

        parser.add_argument(
            "--output",
            type=str,
            default="",
            help="Путь для сохранения JSON-отчёта.",
        )

    def handle(self, *args, **options):
        sizes = options["sizes"]
        repeats = options["repeats"]
        warmup = options["warmup"]
        device_option = options["device"]

        if any(size < 2 for size in sizes):
            raise CommandError(
                "Все размеры матриц должны быть не меньше 2."
            )

        if len(set(sizes)) != len(sizes):
            raise CommandError(
                "Размеры матриц не должны повторяться."
            )

        if repeats < 1:
            raise CommandError(
                "Количество повторов должно быть не меньше 1."
            )

        if warmup < 0:
            raise CommandError(
                "Количество прогревов не может быть отрицательным."
            )

        devices = {
            "both": ("cpu", "cuda"),
            "cpu": ("cpu",),
            "cuda": ("cuda",),
        }[device_option]

        runner = EndToEndBenchmarkRunner()
        reports = []

        self.stdout.write(
            self.style.NOTICE(
                "LAIE: сквозное тестирование CPU/GPU"
            )
        )

        self.stdout.write(
            f"Устройства: {', '.join(devices)}"
        )
        self.stdout.write(f"Размеры: {sizes}")
        self.stdout.write(
            f"Повторы: {repeats}; прогрев: {warmup}"
        )
        self.stdout.write(
            "GPU: учитывается передача данных CPU/GPU."
        )
        self.stdout.write("")

        for size in sizes:
            self.stdout.write(
                self.style.HTTP_INFO(
                    f"Матрицы {size} x {size}"
                )
            )

            try:
                report = runner.run_matmul(
                    size=size,
                    repeats=repeats,
                    warmup=warmup,
                    devices=devices,
                )

            except (EndToEndBenchmarkError, ValueError) as exc:
                raise CommandError(str(exc)) from exc

            reports.append(report)

            for device, result in report["results"].items():
                self.stdout.write(
                    f"  {device.upper()}: "
                    f"полное время "
                    f"{result['median_total_seconds']:.6f} с"
                )

                self.stdout.write(
                    f"    Вычисление: "
                    f"{result['median_compute_seconds']:.6f} с"
                )

                self.stdout.write(
                    f"    Передача CPU -> GPU: "
                    f"{result['median_input_transfer_seconds']:.6f} с"
                )

                self.stdout.write(
                    f"    Передача GPU -> CPU: "
                    f"{result['median_output_transfer_seconds']:.6f} с"
                )

                self.stdout.write(
                    f"    Относительная L2-ошибка: "
                    f"{result['relative_l2_error']:.9g}; "
                    f"корректность: {result['correct']}"
                )

                if device == "cuda":
                    self.stdout.write(
                        f"    Свободная VRAM: "
                        f"{result['free_memory_mb_before']} -> "
                        f"{result['free_memory_mb_after']} МБ"
                    )

            speedup = report["gpu_end_to_end_speedup"]

            if speedup is not None:
                if speedup > 1:
                    self.stdout.write(
                        self.style.SUCCESS(
                            f"  Сквозное ускорение GPU: "
                            f"{speedup:.2f}x"
                        )
                    )
                else:
                    self.stdout.write(
                        self.style.WARNING(
                            f"  GPU не быстрее CPU: "
                            f"{speedup:.2f}x"
                        )
                    )

            for device, reason in report["skipped"].items():
                self.stdout.write(
                    self.style.WARNING(
                        f"  {device.upper()} пропущен: {reason}"
                    )
                )

            self.stdout.write("")

        output = options["output"]

        if output:
            output_path = Path(output).expanduser()

            if not output_path.is_absolute():
                output_path = Path.cwd() / output_path
        else:
            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            output_path = (
                Path.cwd()
                / "reports"
                / "benchmarks"
                / f"end_to_end_{timestamp}.json"
            )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "created_at": datetime.now().astimezone().isoformat(),
            "application": "LAIE",
            "benchmark_type": "end_to_end",
            "reports": reports,
        }

        output_path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        self.stdout.write(
            self.style.SUCCESS(
                f"JSON-отчёт сохранён: {output_path}"
            )
        )
