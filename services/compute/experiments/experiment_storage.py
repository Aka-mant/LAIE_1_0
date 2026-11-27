"""Локальное JSON-хранилище результатов экспериментов LAIE."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from services.compute.experiments.experiment_models import ExperimentResult


class ExperimentStorageError(Exception):
    """Ошибка сохранения или чтения результата эксперимента."""


class ExperimentStorage:
    """Атомарное локальное хранилище JSON-отчётов экспериментов."""

    SCHEMA_VERSION = 1
    DEFAULT_DIRECTORY = Path("reports") / "compute_experiments"

    def __init__(self, directory: Path | str | None = None) -> None:
        """Инициализирует хранилище."""
        self.directory = Path(
            directory if directory is not None else self.DEFAULT_DIRECTORY
        ).resolve()

    def save(self, result: ExperimentResult) -> Path:
        """Атомарно сохраняет результат эксперимента в JSON."""
        if not isinstance(result, ExperimentResult):
            raise TypeError("result должен быть ExperimentResult")

        experiment_id = result.experiment_id

        if not re.fullmatch(r"[A-Za-z0-9_-]+", experiment_id):
            raise ExperimentStorageError(
                "experiment_id содержит недопустимые символы"
            )

        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.directory / f"{experiment_id}.json"

        payload = result.as_dict()
        payload["schema_version"] = self.SCHEMA_VERSION

        temporary_path: Path | None = None

        try:
            serialized = json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )

            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=self.directory,
                prefix=f".{experiment_id}_",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                temporary_file.write(serialized)
                temporary_file.write("\n")
                temporary_file.flush()
                os.fsync(temporary_file.fileno())

            os.replace(temporary_path, target)
            temporary_path = None

        except (OSError, TypeError, ValueError) as exc:
            raise ExperimentStorageError(
                f"Не удалось сохранить эксперимент {experiment_id}: {exc}"
            ) from exc

        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

        return target

    def load(self, path: Path | str) -> dict[str, Any]:
        """Загружает и проверяет JSON-отчёт."""
        source = Path(path)

        if not source.is_absolute():
            source = self.directory / source

        try:
            resolved_source = source.resolve()
            resolved_source.relative_to(self.directory)
        except (OSError, ValueError) as exc:
            raise ExperimentStorageError(
                "Путь отчёта находится за пределами каталога экспериментов"
            ) from exc

        if resolved_source.suffix.lower() != ".json":
            raise ExperimentStorageError(
                "Поддерживаются только JSON-отчёты"
            )

        try:
            with resolved_source.open("r", encoding="utf-8") as file:
                payload = json.load(file)

        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ExperimentStorageError(
                f"Не удалось прочитать отчёт {resolved_source}: {exc}"
            ) from exc

        if not isinstance(payload, dict):
            raise ExperimentStorageError(
                "Корень JSON-отчёта должен быть объектом"
            )

        if payload.get("schema_version") != self.SCHEMA_VERSION:
            raise ExperimentStorageError(
                "Неподдерживаемая версия схемы JSON-отчёта"
            )

        if payload.get("experiment_type") != "prediction_accuracy":
            raise ExperimentStorageError(
                "Файл не является отчётом точности прогнозирования"
            )

        if not isinstance(payload.get("experiment_id"), str):
            raise ExperimentStorageError(
                "В отчёте отсутствует корректный experiment_id"
            )

        return payload

    def list_reports(self) -> tuple[Path, ...]:
        """Возвращает список JSON-отчётов в порядке от новых к старым."""
        if not self.directory.exists():
            return ()

        try:
            reports = (
                path
                for path in self.directory.glob("*.json")
                if path.is_file()
            )
            return tuple(
                sorted(
                    reports,
                    key=lambda path: path.stat().st_mtime,
                    reverse=True,
                )
            )
        except OSError as exc:
            raise ExperimentStorageError(
                f"Не удалось получить список отчётов: {exc}"
            ) from exc
