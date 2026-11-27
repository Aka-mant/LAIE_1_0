
import pytest

from services.compute.benchmark import (
    BenchmarkResult,
    BenchmarkRunner,
)


def test_invalid_matrix_size_is_rejected():
    with pytest.raises(ValueError):
        BenchmarkRunner._validate(1, 3, 1)


def test_invalid_repeat_count_is_rejected():
    with pytest.raises(ValueError):
        BenchmarkRunner._validate(512, 0, 1)


def test_negative_warmup_is_rejected():
    with pytest.raises(ValueError):
        BenchmarkRunner._validate(512, 3, -1)


def test_valid_parameters_are_accepted():
    BenchmarkRunner._validate(512, 3, 1)


def test_benchmark_result_is_serializable():
    result = BenchmarkResult(
        device="cpu",
        size=512,
        repeats=3,
        warmup=1,
        median_seconds=0.01,
        min_seconds=0.009,
        max_seconds=0.012,
        max_absolute_error=0.0,
        correct=True,
    )

    data = result.as_dict()

    assert data["device"] == "cpu"
    assert data["size"] == 512
    assert data["repeats"] == 3
    assert data["correct"] is True


def test_invalid_device_is_rejected():
    runner = BenchmarkRunner()

    with pytest.raises(ValueError):
        runner.run_matmul(
            size=512,
            repeats=1,
            warmup=0,
            devices=("tpu",),
        )


def test_duplicate_devices_are_rejected():
    runner = BenchmarkRunner()

    with pytest.raises(ValueError):
        runner.run_matmul(
            size=512,
            repeats=1,
            warmup=0,
            devices=("cpu", "cpu"),
        )
