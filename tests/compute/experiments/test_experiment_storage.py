
"""Тесты локального JSON-хранилища экспериментов."""

import json
from pathlib import Path

import pytest

from services.compute.experiments.experiment_models import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
)
from services.compute.experiments.experiment_storage import (
    ExperimentStorage,
    ExperimentStorageError,
)


@pytest.fixture
def storage(tmp_path: Path) -> ExperimentStorage:
    """Создаёт изолированное временное хранилище."""
    return ExperimentStorage(tmp_path / "experiments")


@pytest.fixture
def experiment_result() -> ExperimentResult:
    """Создаёт минимальный корректный результат эксперимента."""
    return ExperimentResult(
        experiment_id="test_experiment_001",
        created_at="2026-11-25T16:00:00+00:00",
        status=ExperimentStatus.COMPLETED,
        config=ExperimentConfig(
            calibration_sizes=(256,),
            validation_sizes=(512,),
            batch_sizes=(64,),
            repeats=2,
            cuda_warmup=0,
            high_confidence_samples=2,
        ),
        hardware={"cpu": "Test CPU", "cuda_available": False},
        software={"python": "3.14"},
        cases=(),
        predictions=(),
        measurements=(),
        comparisons=(),
        decisions=(),
        calibration_summary=(),
    )


def test_save_creates_json_report(
    storage: ExperimentStorage,
    experiment_result: ExperimentResult,
) -> None:
    """Сохранение создаёт JSON-файл с корректной схемой."""
    report_path = storage.save(experiment_result)

    assert report_path.exists()
    assert report_path.name == "test_experiment_001.json"

    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == ExperimentStorage.SCHEMA_VERSION
    assert payload["experiment_id"] == experiment_result.experiment_id
    assert payload["experiment_type"] == "prediction_accuracy"
    assert payload["status"] == "completed"


def test_save_uses_utf8_and_preserves_unicode(
    storage: ExperimentStorage,
    experiment_result: ExperimentResult,
) -> None:
    """JSON сохраняет Unicode без экранирования кириллицы."""
    result = ExperimentResult(
        experiment_id=experiment_result.experiment_id,
        created_at=experiment_result.created_at,
        status=experiment_result.status,
        config=experiment_result.config,
        hardware={"processor": "Тестовый процессор"},
        software=experiment_result.software,
        cases=(),
        predictions=(),
        measurements=(),
        comparisons=(),
        decisions=(),
        calibration_summary=(),
    )

    report_path = storage.save(result)

    content = report_path.read_text(encoding="utf-8")

    assert "Тестовый процессор" in content


def test_load_returns_validated_payload(
    storage: ExperimentStorage,
    experiment_result: ExperimentResult,
) -> None:
    """Чтение возвращает проверенный JSON-словарь."""
    report_path = storage.save(experiment_result)

    payload = storage.load(report_path)

    assert payload["experiment_id"] == experiment_result.experiment_id
    assert payload["schema_version"] == 1
    assert payload["config"]["dtype"] == "float64"


def test_load_accepts_filename_relative_to_storage(
    storage: ExperimentStorage,
    experiment_result: ExperimentResult,
) -> None:
    """Чтение принимает имя файла относительно каталога."""
    storage.save(experiment_result)

    payload = storage.load("test_experiment_001.json")

    assert payload["experiment_id"] == experiment_result.experiment_id


def test_load_rejects_path_outside_storage(
    storage: ExperimentStorage,
    tmp_path: Path,
) -> None:
    """Чтение не позволяет выйти за пределы каталога."""
    outside_file = tmp_path / "outside.json"
    outside_file.write_text("{}", encoding="utf-8")

    with pytest.raises(
        ExperimentStorageError,
        match="за пределами каталога",
    ):
        storage.load(outside_file)


def test_load_rejects_non_json_file(
    storage: ExperimentStorage,
    tmp_path: Path,
) -> None:
    """Чтение отклоняет файлы с другим расширением."""
    text_file = storage.directory / "report.txt"
    storage.directory.mkdir(parents=True)
    text_file.write_text("report", encoding="utf-8")

    with pytest.raises(
        ExperimentStorageError,
        match="только JSON",
    ):
        storage.load(text_file)


def test_load_rejects_invalid_json(
    storage: ExperimentStorage,
) -> None:
    """Чтение отклоняет повреждённый JSON."""
    storage.directory.mkdir(parents=True)
    report_path = storage.directory / "broken.json"
    report_path.write_text("{invalid json", encoding="utf-8")

    with pytest.raises(ExperimentStorageError, match="Не удалось прочитать"):
        storage.load(report_path)


def test_load_rejects_unsupported_schema(
    storage: ExperimentStorage,
) -> None:
    """Чтение отклоняет неизвестную версию схемы."""
    storage.directory.mkdir(parents=True)
    report_path = storage.directory / "future.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": 999,
                "experiment_type": "prediction_accuracy",
                "experiment_id": "future",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ExperimentStorageError,
        match="версия схемы",
    ):
        storage.load(report_path)


def test_load_rejects_other_experiment_type(
    storage: ExperimentStorage,
) -> None:
    """Чтение отклоняет отчёт другого типа."""
    storage.directory.mkdir(parents=True)
    report_path = storage.directory / "other.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": ExperimentStorage.SCHEMA_VERSION,
                "experiment_type": "legacy_benchmark",
                "experiment_id": "other",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ExperimentStorageError,
        match="не является отчётом",
    ):
        storage.load(report_path)


@pytest.mark.parametrize(
    "experiment_id",
    [
        "../outside",
        "nested/report",
        "report name",
        "report.json",
    ],
)
def test_save_rejects_invalid_experiment_id(
    storage: ExperimentStorage,
    experiment_result: ExperimentResult,
    experiment_id: str,
) -> None:
    """Идентификатор не может содержать путь или спецсимволы."""
    result = ExperimentResult(
        experiment_id=experiment_id,
        created_at=experiment_result.created_at,
        status=experiment_result.status,
        config=experiment_result.config,
        hardware=experiment_result.hardware,
        software=experiment_result.software,
        cases=(),
        predictions=(),
        measurements=(),
        comparisons=(),
        decisions=(),
        calibration_summary=(),
    )

    with pytest.raises(
        ExperimentStorageError,
        match="недопустимые символы",
    ):
        storage.save(result)


def test_save_rejects_wrong_result_type(
    storage: ExperimentStorage,
) -> None:
    """Сохранение требует ExperimentResult."""
    with pytest.raises(TypeError, match="ExperimentResult"):
        storage.save({"experiment_id": "invalid"})  # type: ignore[arg-type]


def test_list_reports_returns_empty_tuple_when_missing(
    storage: ExperimentStorage,
) -> None:
    """Отсутствующий каталог не считается ошибкой."""
    assert storage.list_reports() == ()


def test_list_reports_returns_json_files_only(
    storage: ExperimentStorage,
    experiment_result: ExperimentResult,
) -> None:
    """Список содержит только JSON-отчёты."""
    storage.save(experiment_result)
    (storage.directory / "temporary.tmp").write_text(
        "temporary",
        encoding="utf-8",
    )

    reports = storage.list_reports()

    assert len(reports) == 1
    assert reports[0].suffix == ".json"


def test_save_does_not_leave_temporary_files(
    storage: ExperimentStorage,
    experiment_result: ExperimentResult,
) -> None:
    """После успешной записи временный файл удалён."""
    storage.save(experiment_result)

    temporary_files = tuple(storage.directory.glob("*.tmp"))

    assert temporary_files == ()


def test_save_replaces_previous_report_atomically(
    storage: ExperimentStorage,
    experiment_result: ExperimentResult,
) -> None:
    """Повторное сохранение обновляет тот же отчёт."""
    first_path = storage.save(experiment_result)

    updated = ExperimentResult(
        experiment_id=experiment_result.experiment_id,
        created_at="2026-11-25T17:00:00+00:00",
        status=ExperimentStatus.PARTIAL,
        config=experiment_result.config,
        hardware=experiment_result.hardware,
        software=experiment_result.software,
        cases=(),
        predictions=(),
        measurements=(),
        comparisons=(),
        decisions=(),
        calibration_summary=(),
    )

    second_path = storage.save(updated)

    assert first_path == second_path
    payload = storage.load(second_path)
    assert payload["status"] == "partial"
