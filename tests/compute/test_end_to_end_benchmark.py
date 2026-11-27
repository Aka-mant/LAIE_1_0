
"""Тесты сквозного бенчмарка LAIE."""

from unittest.mock import Mock

import pytest

from services.compute.end_to_end_benchmark import (
    EndToEndBenchmarkRunner,
)


def test_invalid_matrix_size_is_rejected():
    with pytest.raises(ValueError):
        EndToEndBenchmarkRunner._validate(1, 3, 1)


def test_invalid_repeat_count_is_rejected():
    with pytest.raises(ValueError):
        EndToEndBenchmarkRunner._validate(512, 0, 1)


def test_negative_warmup_is_rejected():
    with pytest.raises(ValueError):
        EndToEndBenchmarkRunner._validate(512, 3, -1)


def test_valid_parameters_are_accepted():
    EndToEndBenchmarkRunner._validate(512, 3, 1)


def test_invalid_device_is_rejected():
    runner = EndToEndBenchmarkRunner()

    with pytest.raises(ValueError):
        runner.run_matmul(
            size=8,
            repeats=1,
            warmup=0,
            devices=("tpu",),
        )


def test_duplicate_devices_are_rejected():
    runner = EndToEndBenchmarkRunner()

    with pytest.raises(ValueError):
        runner.run_matmul(
            size=8,
            repeats=1,
            warmup=0,
            devices=("cpu", "cpu"),
        )


def test_accuracy_accepts_identical_results():
    torch = pytest.importorskip("torch")

    reference = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        dtype=torch.float64,
    )

    result = EndToEndBenchmarkRunner._measure_accuracy(
        torch=torch,
        actual=reference.clone(),
        reference=reference,
    )

    assert result["correct"] is True
    assert result["max_absolute_error"] == 0.0
    assert result["rmse"] == 0.0
    assert result["relative_l2_error"] == 0.0


def test_accuracy_rejects_non_finite_results():
    torch = pytest.importorskip("torch")

    reference = torch.ones(
        (2, 2),
        dtype=torch.float64,
    )

    actual = torch.tensor(
        [[float("inf"), 1.0], [1.0, 1.0]],
        dtype=torch.float64,
    )

    result = EndToEndBenchmarkRunner._measure_accuracy(
        torch=torch,
        actual=actual,
        reference=reference,
    )

    assert result["correct"] is False
    assert result["relative_l2_error"] == float("inf")


def test_accuracy_rejects_large_relative_error():
    torch = pytest.importorskip("torch")

    reference = torch.ones(
        (4, 4),
        dtype=torch.float64,
    )

    actual = reference * 2.0

    result = EndToEndBenchmarkRunner._measure_accuracy(
        torch=torch,
        actual=actual,
        reference=reference,
    )

    assert result["correct"] is False
    assert (
        result["relative_l2_error"]
        > EndToEndBenchmarkRunner.MAX_RELATIVE_L2_ERROR
    )


def test_cpu_end_to_end_benchmark():
    pytest.importorskip("torch")

    runner = EndToEndBenchmarkRunner()

    report = runner.run_matmul(
        size=32,
        repeats=2,
        warmup=1,
        devices=("cpu",),
    )

    assert report["benchmark"] == "end_to_end_matmul_float32"
    assert report["size"] == 32
    assert "cpu" in report["results"]

    result = report["results"]["cpu"]

    assert result["device"] == "cpu"
    assert result["correct"] is True
    assert result["median_total_seconds"] >= 0
    assert result["median_compute_seconds"] >= 0
    assert result["median_input_transfer_seconds"] == 0
    assert result["median_output_transfer_seconds"] == 0

    assert result["relative_l2_error"] <= (
        EndToEndBenchmarkRunner.MAX_RELATIVE_L2_ERROR
    )

    assert report["gpu_end_to_end_speedup"] is None


def test_unavailable_cuda_is_reported_as_skipped():
    pytest.importorskip("torch")

    engine = Mock()
    engine.inspect.return_value = {
        "cuda_available": False,
    }

    runner = EndToEndBenchmarkRunner(
        engine=engine,
    )

    report = runner.run_matmul(
        size=16,
        repeats=1,
        warmup=0,
        devices=("cpu", "cuda"),
    )

    assert "cpu" in report["results"]
    assert "cuda" not in report["results"]
    assert report["skipped"]["cuda"] == "CUDA недоступна."


def test_cpu_only_benchmark_does_not_inspect_gpu():
    pytest.importorskip("torch")

    engine = Mock()
    runner = EndToEndBenchmarkRunner(
        engine=engine,
    )

    runner.run_matmul(
        size=16,
        repeats=1,
        warmup=0,
        devices=("cpu",),
    )

    engine.inspect.assert_not_called()
    engine.explain_choice.assert_not_called()
