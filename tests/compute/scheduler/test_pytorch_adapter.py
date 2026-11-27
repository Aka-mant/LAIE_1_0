
"""Тесты PyTorch-адаптера CPU/CUDA."""

from __future__ import annotations

import pytest

from services.compute.scheduler.pytorch_adapter import (
    BackendUnavailableError,
    CUDAExecutionError,
    InvalidTensorError,
    MatmulExecution,
    PyTorchCUDAAdapter,
)
from services.compute.scheduler.task_types import ComputeBackend

torch = pytest.importorskip("torch")


@pytest.fixture
def adapter() -> PyTorchCUDAAdapter:
    return PyTorchCUDAAdapter(torch_module=torch)


def test_cuda_availability_matches_torch(adapter: PyTorchCUDAAdapter) -> None:
    assert adapter.is_cuda_available() is torch.cuda.is_available()


def test_cpu_matmul_returns_correct_result(
    adapter: PyTorchCUDAAdapter,
) -> None:
    left = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        dtype=torch.float64,
    )
    right = torch.tensor(
        [[5.0, 6.0], [7.0, 8.0]],
        dtype=torch.float64,
    )

    execution = adapter.matmul(
        left,
        right,
        backend=ComputeBackend.CPU,
    )

    expected = torch.tensor(
        [[19.0, 22.0], [43.0, 50.0]],
        dtype=torch.float64,
    )

    assert torch.equal(execution.result, expected)
    assert execution.backend is ComputeBackend.CPU
    assert execution.compute_seconds > 0
    assert execution.transfer_seconds == 0
    assert execution.total_seconds >= execution.compute_seconds
    assert execution.peak_memory_bytes is None


def test_cpu_matmul_calculates_operation_count(
    adapter: PyTorchCUDAAdapter,
) -> None:
    left = torch.ones((3, 4), dtype=torch.float32)
    right = torch.ones((4, 5), dtype=torch.float32)

    execution = adapter.matmul(left, right)

    assert execution.operations == 2 * 3 * 4 * 5
    assert execution.input_bytes == (
        left.numel() * left.element_size()
        + right.numel() * right.element_size()
    )
    assert execution.output_bytes == (
        execution.result.numel() * execution.result.element_size()
    )


def test_cpu_matmul_supports_float32(
    adapter: PyTorchCUDAAdapter,
) -> None:
    left = torch.ones((2, 3), dtype=torch.float32)
    right = torch.ones((3, 2), dtype=torch.float32)

    execution = adapter.matmul(left, right)

    assert execution.result.dtype is torch.float32
    assert torch.equal(
        execution.result,
        torch.full((2, 2), 3.0, dtype=torch.float32),
    )


def test_cpu_matmul_supports_float64(
    adapter: PyTorchCUDAAdapter,
) -> None:
    left = torch.ones((2, 3), dtype=torch.float64)
    right = torch.ones((3, 2), dtype=torch.float64)

    execution = adapter.matmul(left, right)

    assert execution.result.dtype is torch.float64


def test_rejects_non_tensor_input(
    adapter: PyTorchCUDAAdapter,
) -> None:
    with pytest.raises(InvalidTensorError, match="torch.Tensor"):
        adapter.matmul([[1.0]], torch.ones((1, 1)))


def test_rejects_non_2d_tensor(
    adapter: PyTorchCUDAAdapter,
) -> None:
    left = torch.ones((2, 2, 2))
    right = torch.ones((2, 2))

    with pytest.raises(InvalidTensorError, match="двумерными"):
        adapter.matmul(left, right)


def test_rejects_incompatible_shapes(
    adapter: PyTorchCUDAAdapter,
) -> None:
    left = torch.ones((2, 3))
    right = torch.ones((4, 2))

    with pytest.raises(InvalidTensorError, match="столбцов"):
        adapter.matmul(left, right)


def test_rejects_different_dtypes(
    adapter: PyTorchCUDAAdapter,
) -> None:
    left = torch.ones((2, 2), dtype=torch.float32)
    right = torch.ones((2, 2), dtype=torch.float64)

    with pytest.raises(InvalidTensorError, match="одинаковый dtype"):
        adapter.matmul(left, right)


def test_rejects_integer_dtype(
    adapter: PyTorchCUDAAdapter,
) -> None:
    left = torch.ones((2, 2), dtype=torch.int64)
    right = torch.ones((2, 2), dtype=torch.int64)

    with pytest.raises(InvalidTensorError, match="float32"):
        adapter.matmul(left, right)


def test_rejects_non_backend_argument(
    adapter: PyTorchCUDAAdapter,
) -> None:
    with pytest.raises(TypeError, match="ComputeBackend"):
        adapter.matmul(
            torch.ones((2, 2)),
            torch.ones((2, 2)),
            backend="cuda",
        )


def test_rejects_cuda_input_tensor(
    adapter: PyTorchCUDAAdapter,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA недоступна.")

    left = torch.ones((2, 2), device="cuda")
    right = torch.ones((2, 2))

    with pytest.raises(InvalidTensorError, match="на CPU"):
        adapter.matmul(left, right)


def test_cuda_matmul_returns_result_on_cpu(
    adapter: PyTorchCUDAAdapter,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA недоступна.")

    left = torch.randn((256, 256), dtype=torch.float32)
    right = torch.randn((256, 256), dtype=torch.float32)

    execution = adapter.matmul(
        left,
        right,
        backend=ComputeBackend.CUDA,
    )

    expected = torch.matmul(left, right)

    assert execution.backend is ComputeBackend.CUDA
    assert execution.result.device.type == "cpu"
    assert torch.allclose(
        execution.result,
        expected,
        rtol=1e-4,
        atol=1e-4,
    )
    assert execution.compute_seconds > 0
    assert execution.transfer_seconds >= 0
    assert execution.total_seconds >= (
        execution.compute_seconds + execution.transfer_seconds
    )
    assert execution.peak_memory_bytes is not None
    assert execution.peak_memory_bytes > 0


def test_cuda_matmul_uses_float64(
    adapter: PyTorchCUDAAdapter,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA недоступна.")

    left = torch.randn((32, 32), dtype=torch.float64)
    right = torch.randn((32, 32), dtype=torch.float64)

    execution = adapter.matmul(
        left,
        right,
        backend=ComputeBackend.CUDA,
    )

    expected = torch.matmul(left, right)

    assert execution.result.dtype is torch.float64
    assert torch.allclose(
        execution.result,
        expected,
        rtol=1e-10,
        atol=1e-10,
    )


def test_cuda_backend_rejects_unavailable_device() -> None:
    class UnavailableCUDA:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeTorch:
        Tensor = torch.Tensor
        float32 = torch.float32
        float64 = torch.float64
        cuda = UnavailableCUDA()

    adapter = PyTorchCUDAAdapter(torch_module=FakeTorch)

    left = torch.ones((2, 2))
    right = torch.ones((2, 2))

    with pytest.raises(
        BackendUnavailableError,
        match="CUDA недоступна",
    ):
        adapter.matmul(
            left,
            right,
            backend=ComputeBackend.CUDA,
        )


def test_cuda_execution_error_is_wrapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenCUDA:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def current_device() -> int:
            return 0

        @staticmethod
        def synchronize(device: object) -> None:
            raise RuntimeError("simulated CUDA failure")

    class FakeTorch:
        Tensor = torch.Tensor
        float32 = torch.float32
        float64 = torch.float64
        cuda = BrokenCUDA()

        @staticmethod
        def device(name: str) -> str:
            return name

    adapter = PyTorchCUDAAdapter(torch_module=FakeTorch)

    left = torch.ones((2, 2))
    right = torch.ones((2, 2))

    with pytest.raises(CUDAExecutionError, match="simulated CUDA failure"):
        adapter.matmul(
            left,
            right,
            backend=ComputeBackend.CUDA,
        )


def test_matmul_execution_rejects_invalid_metrics() -> None:
    with pytest.raises(ValueError, match="compute_seconds"):
        MatmulExecution(
            result=None,
            backend=ComputeBackend.CPU,
            compute_seconds=0,
            transfer_seconds=0,
            total_seconds=1,
            peak_memory_bytes=None,
            operations=1,
            input_bytes=1,
            output_bytes=1,
        )


def test_matmul_execution_rejects_cpu_transfer_time() -> None:
    with pytest.raises(ValueError, match="CPU"):
        MatmulExecution(
            result=None,
            backend=ComputeBackend.CPU,
            compute_seconds=1,
            transfer_seconds=1,
            total_seconds=2,
            peak_memory_bytes=None,
            operations=1,
            input_bytes=1,
            output_bytes=1,
        )


def test_matmul_execution_as_dict() -> None:
    execution = MatmulExecution(
        result=None,
        backend=ComputeBackend.CPU,
        compute_seconds=0.5,
        transfer_seconds=0,
        total_seconds=0.6,
        peak_memory_bytes=None,
        operations=100,
        input_bytes=200,
        output_bytes=100,
    )

    data = execution.as_dict()

    assert data["backend"] == "cpu"
    assert data["compute_seconds"] == 0.5
    assert data["operations"] == 100
    assert "result" not in data
