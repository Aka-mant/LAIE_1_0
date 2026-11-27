
from types import SimpleNamespace

import pytest

from services.compute.engine import ComputeEngine


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        ComputeEngine(min_gpu_items=0)

    with pytest.raises(ValueError):
        ComputeEngine(memory_reserve_mb=-1)


def test_cpu_can_be_forced_without_loading_gpu():
    engine = ComputeEngine()

    assert engine.choose_device(10**9, force="cpu") == "cpu"
    assert engine._torch_checked is False


def test_invalid_force_is_rejected():
    engine = ComputeEngine()

    with pytest.raises(ValueError):
        engine.choose_device(10, force="tpu")


def test_negative_task_size_is_rejected():
    engine = ComputeEngine()

    with pytest.raises(ValueError):
        engine.choose_device(-1)


def test_negative_memory_estimate_is_rejected():
    engine = ComputeEngine()

    with pytest.raises(ValueError):
        engine.choose_device(100, estimated_memory_mb=-1)

    with pytest.raises(ValueError):
        engine.choose_device(100, workspace_memory_mb=-1)


def test_cpu_is_selected_when_cuda_is_unavailable(monkeypatch):
    engine = ComputeEngine()

    monkeypatch.setattr(
        engine,
        "inspect",
        lambda: {"cuda_available": False},
    )

    decision = engine.explain_choice(1_000_000)

    assert decision.device == "cpu"
    assert "CUDA недоступна" in decision.reason


def test_cpu_is_selected_when_gpu_memory_is_insufficient(monkeypatch):
    engine = ComputeEngine(memory_reserve_mb=1_536)

    monkeypatch.setattr(
        engine,
        "inspect",
        lambda: {
            "cuda_available": True,
            "gpu": {"free_memory_mb": 2_000},
        },
    )

    decision = engine.explain_choice(
        item_count=1_000_000,
        estimated_memory_mb=600,
    )

    assert decision.device == "cpu"
    assert decision.required_memory_mb == 600
    assert decision.available_memory_mb == 2_000
    assert "Недостаточно свободной VRAM" in decision.reason


def test_workspace_memory_is_included(monkeypatch):
    engine = ComputeEngine(memory_reserve_mb=1_000)

    monkeypatch.setattr(
        engine,
        "inspect",
        lambda: {
            "cuda_available": True,
            "gpu": {"free_memory_mb": 2_000},
        },
    )

    decision = engine.explain_choice(
        item_count=1_000_000,
        estimated_memory_mb=300,
        workspace_memory_mb=300,
    )

    assert decision.device == "cuda"
    assert decision.required_memory_mb == 600


def test_workspace_can_force_cpu_fallback(monkeypatch):
    engine = ComputeEngine(memory_reserve_mb=1_000)

    monkeypatch.setattr(
        engine,
        "inspect",
        lambda: {
            "cuda_available": True,
            "gpu": {"free_memory_mb": 2_000},
        },
    )

    decision = engine.explain_choice(
        item_count=1_000_000,
        estimated_memory_mb=300,
        workspace_memory_mb=800,
    )

    assert decision.device == "cpu"
    assert decision.required_memory_mb == 1_100


def test_small_task_uses_cpu(monkeypatch):
    engine = ComputeEngine(min_gpu_items=100_000)

    monkeypatch.setattr(
        engine,
        "inspect",
        lambda: {
            "cuda_available": True,
            "gpu": {"free_memory_mb": 7_000},
        },
    )

    decision = engine.explain_choice(1_000)

    assert decision.device == "cpu"
    assert "меньше порога GPU" in decision.reason


def test_large_task_uses_cuda(monkeypatch):
    engine = ComputeEngine(
        min_gpu_items=100_000,
        memory_reserve_mb=1_000,
    )

    monkeypatch.setattr(
        engine,
        "inspect",
        lambda: {
            "cuda_available": True,
            "gpu": {"free_memory_mb": 7_000},
        },
    )

    decision = engine.explain_choice(1_000_000)

    assert decision.device == "cuda"


def test_forced_cuda_still_checks_memory(monkeypatch):
    engine = ComputeEngine(memory_reserve_mb=1_000)

    monkeypatch.setattr(
        engine,
        "inspect",
        lambda: {
            "cuda_available": True,
            "gpu": {"free_memory_mb": 1_200},
        },
    )

    decision = engine.explain_choice(
        item_count=1_000_000,
        estimated_memory_mb=300,
        force="cuda",
    )

    assert decision.device == "cpu"
    assert "Недостаточно свободной VRAM" in decision.reason


def test_last_decision_is_saved(monkeypatch):
    engine = ComputeEngine()

    monkeypatch.setattr(
        engine,
        "inspect",
        lambda: {
            "cuda_available": True,
            "gpu": {"free_memory_mb": 7_000},
        },
    )

    engine.choose_device(1_000_000)

    assert engine.last_decision is not None
    assert engine.last_decision.device == "cuda"


def test_cuda_failure_falls_back_to_cpu(monkeypatch):
    class FakeTensor:
        ndim = 2
        shape = (2, 2)
        device = SimpleNamespace(type="cpu")

        def numel(self):
            return 4

        def element_size(self):
            return 4

        def to(self, device):
            if device == "cuda":
                raise RuntimeError("Имитированная ошибка CUDA")
            return self

        def __matmul__(self, other):
            return "cpu-result"

    class FakeCuda:
        @staticmethod
        def empty_cache():
            pass

    fake_torch = SimpleNamespace(cuda=FakeCuda())

    engine = ComputeEngine()

    monkeypatch.setattr(engine, "_load_torch", lambda: fake_torch)

    monkeypatch.setattr(
        engine,
        "explain_choice",
        lambda **kwargs: SimpleNamespace(
            device="cuda",
            required_memory_mb=1,
            available_memory_mb=100,
        ),
    )

    left = FakeTensor()
    right = FakeTensor()

    result = engine.tensor_matmul(left, right, force="cuda")

    assert result == "cpu-result"
    assert engine.last_decision.device == "cpu"
    assert "Ошибка CUDA" in engine.last_decision.reason



def test_dynamic_vram_reserve_uses_percentage_of_current_free_memory(monkeypatch):
    engine = ComputeEngine(
        memory_reserve_mb=0,
        memory_reserve_ratio=0.30,
    )
    monkeypatch.setattr(
        engine,
        "inspect",
        lambda: {
            "cuda_available": True,
            "gpu": {"free_memory_mb": 1_000},
        },
    )

    fits_at_reserve_boundary = engine.explain_choice(
        item_count=1_000_000,
        estimated_memory_mb=700,
    )
    exceeds_reserve_boundary = engine.explain_choice(
        item_count=1_000_000,
        estimated_memory_mb=701,
    )

    assert fits_at_reserve_boundary.device == "cuda"
    assert exceeds_reserve_boundary.device == "cpu"
    assert "резерва (300 МБ)" in exceeds_reserve_boundary.reason


def test_invalid_dynamic_vram_reserve_ratio_is_rejected():
    with pytest.raises(ValueError):
        ComputeEngine(memory_reserve_ratio=1.0)

    with pytest.raises(ValueError):
        ComputeEngine(memory_reserve_ratio=-0.1)

    with pytest.raises(TypeError):
        ComputeEngine(memory_reserve_ratio=True)


def test_task_and_memory_sizes_require_integers():
    engine = ComputeEngine()

    with pytest.raises(TypeError):
        engine.choose_device(True)

    with pytest.raises(TypeError):
        engine.choose_device(10.5)

    with pytest.raises(TypeError):
        engine.choose_device(10, estimated_memory_mb=1.5)


def test_missing_gpu_memory_information_falls_back_to_cpu(monkeypatch):
    engine = ComputeEngine()
    monkeypatch.setattr(
        engine,
        "inspect",
        lambda: {
            "cuda_available": True,
            "gpu": {"free_memory_mb": None},
        },
    )

    decision = engine.explain_choice(1_000_000)

    assert decision.device == "cpu"
    assert "не удалось достоверно определить" in decision.reason.casefold()


def test_tensor_matmul_rounds_memory_estimates_up(monkeypatch):
    class FakeTensor:
        ndim = 2
        shape = (1, 1)
        device = SimpleNamespace(type="cpu")

        def numel(self):
            return 1

        def element_size(self):
            return 1

        def to(self, device):
            return self

        def __matmul__(self, other):
            return "cpu-result"

    engine = ComputeEngine()
    monkeypatch.setattr(
        engine,
        "_load_torch",
        lambda: SimpleNamespace(cuda=SimpleNamespace()),
    )

    captured = {}

    def capture_decision(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(device="cpu")

    monkeypatch.setattr(engine, "explain_choice", capture_decision)

    result = engine.tensor_matmul(FakeTensor(), FakeTensor())

    assert result == "cpu-result"
    assert captured["estimated_memory_mb"] == 1
    assert captured["workspace_memory_mb"] == 1



@pytest.mark.parametrize(
    ("memory_bytes", "expected_mib"),
    [
        (0, 0),
        (1, 1),
        (1024 * 1024, 1),
        (1024 * 1024 + 1, 2),
    ],
)
def test_bytes_to_mb_ceil_is_conservative(memory_bytes, expected_mib):
    assert ComputeEngine.bytes_to_mb_ceil(memory_bytes) == expected_mib


def test_bytes_to_mb_ceil_rejects_invalid_values():
    with pytest.raises(TypeError):
        ComputeEngine.bytes_to_mb_ceil(True)

    with pytest.raises(ValueError):
        ComputeEngine.bytes_to_mb_ceil(-1)
