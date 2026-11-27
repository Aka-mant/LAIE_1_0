
"""Тесты локального мониторинга оборудования."""

from __future__ import annotations

import pytest


import services.compute.scheduler.hardware_monitor as hardware_monitor_module







from types import SimpleNamespace



from services.compute.scheduler.hardware_monitor import (
    HardwareMonitor,
    HardwareSnapshot,
)
from services.compute.scheduler.task_types import ComputeBackend


class FakeTorchCUDA:
    """Имитация интерфейса torch.cuda."""

    def __init__(
        self,
        *,
        available: bool = True,
        name: str = "Test NVIDIA GPU",
        total: int = 8 * 1024**3,
        free: int = 6 * 1024**3,
    ) -> None:
        self.available = available
        self.name = name
        self.total = total
        self.free = free

    def is_available(self) -> bool:
        return self.available

    @staticmethod
    def current_device() -> int:
        return 0

    def get_device_properties(self, index: int) -> SimpleNamespace:
        assert index == 0
        return SimpleNamespace(name=self.name)

    def mem_get_info(self, index: int) -> tuple[int, int]:
        assert index == 0
        return self.free, self.total


def make_torch(cuda: FakeTorchCUDA) -> SimpleNamespace:
    return SimpleNamespace(cuda=cuda)


def test_hardware_snapshot_accepts_valid_cpu_only_data() -> None:
    snapshot = HardwareSnapshot(
        cpu_name="Test CPU",
        cpu_logical_cores=8,
        ram_total_bytes=16_000,
        ram_available_bytes=8_000,
        cuda_available=False,
    )

    assert snapshot.cpu_name == "Test CPU"
    assert snapshot.cpu_logical_cores == 8
    assert snapshot.cuda_available is False
    assert snapshot.cuda_device_name is None


def test_hardware_snapshot_rejects_zero_cpu_cores() -> None:
    with pytest.raises(ValueError, match="cpu_logical_cores"):
        HardwareSnapshot(
            cpu_name="Test CPU",
            cpu_logical_cores=0,
            ram_total_bytes=16_000,
            ram_available_bytes=8_000,
            cuda_available=False,
        )


def test_hardware_snapshot_rejects_invalid_ram() -> None:
    with pytest.raises(ValueError, match="доступной RAM"):
        HardwareSnapshot(
            cpu_name="Test CPU",
            cpu_logical_cores=8,
            ram_total_bytes=16_000,
            ram_available_bytes=20_000,
            cuda_available=False,
        )


def test_hardware_snapshot_requires_cuda_device_name() -> None:
    with pytest.raises(ValueError, match="имя устройства"):
        HardwareSnapshot(
            cpu_name="Test CPU",
            cpu_logical_cores=8,
            ram_total_bytes=16_000,
            ram_available_bytes=8_000,
            cuda_available=True,
            cuda_total_bytes=8_000,
            cuda_free_bytes=4_000,
        )


def test_inspect_returns_cpu_and_cuda_information(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = make_torch(FakeTorchCUDA())

    monitor = HardwareMonitor(torch_module=torch)

    monkeypatch.setattr(
        HardwareMonitor,
        "_get_system_memory",
        staticmethod(lambda: (32_000, 20_000)),
    )
    monkeypatch.setattr(
        "services.compute.scheduler.hardware_monitor.platform.processor",
        lambda: "Test CPU",
    )
    monkeypatch.setattr(
        "services.compute.scheduler.hardware_monitor.os.cpu_count",
        lambda: 12,
    )

    snapshot = monitor.inspect()

    assert snapshot.cpu_name == "Test CPU"
    assert snapshot.cpu_logical_cores == 12
    assert snapshot.ram_total_bytes == 32_000
    assert snapshot.ram_available_bytes == 20_000

    assert snapshot.cuda_available is True
    assert snapshot.cuda_device_name == "Test NVIDIA GPU"
    assert snapshot.cuda_total_bytes == 8 * 1024**3
    assert snapshot.cuda_free_bytes == 6 * 1024**3


def test_inspect_handles_cuda_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = make_torch(FakeTorchCUDA(available=False))
    monitor = HardwareMonitor(torch_module=torch)

    monkeypatch.setattr(
        HardwareMonitor,
        "_get_system_memory",
        staticmethod(lambda: (16_000, 8_000)),
    )

    snapshot = monitor.inspect()

    assert snapshot.cuda_available is False
    assert snapshot.cuda_device_name is None
    assert snapshot.cuda_total_bytes is None
    assert snapshot.cuda_free_bytes is None
    assert snapshot.cuda_reason


def test_inspect_handles_cuda_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenCUDA:
        @staticmethod
        def is_available() -> bool:
            raise RuntimeError("CUDA initialization failed")

    monitor = HardwareMonitor(
        torch_module=SimpleNamespace(cuda=BrokenCUDA())
    )

    monkeypatch.setattr(
        HardwareMonitor,
        "_get_system_memory",
        staticmethod(lambda: (16_000, 8_000)),
    )

    snapshot = monitor.inspect()

    assert snapshot.cuda_available is False
    assert "CUDA initialization failed" in snapshot.cuda_reason


def test_memory_snapshots_include_cpu_and_cuda() -> None:
    snapshot = HardwareSnapshot(
        cpu_name="Test CPU",
        cpu_logical_cores=8,
        ram_total_bytes=32_000,
        ram_available_bytes=20_000,
        cuda_available=True,
        cuda_device_name="Test GPU",
        cuda_total_bytes=8_000,
        cuda_free_bytes=6_000,
    )

    snapshots = snapshot.memory_snapshots()

    assert len(snapshots) == 2
    assert snapshots[0].backend is ComputeBackend.CPU
    assert snapshots[0].total_bytes == 32_000
    assert snapshots[0].free_bytes == 20_000

    assert snapshots[1].backend is ComputeBackend.CUDA
    assert snapshots[1].total_bytes == 8_000
    assert snapshots[1].free_bytes == 6_000


def test_memory_snapshots_include_only_cpu_without_cuda() -> None:
    snapshot = HardwareSnapshot(
        cpu_name="Test CPU",
        cpu_logical_cores=8,
        ram_total_bytes=32_000,
        ram_available_bytes=20_000,
        cuda_available=False,
    )

    snapshots = snapshot.memory_snapshots()

    assert len(snapshots) == 1
    assert snapshots[0].backend is ComputeBackend.CPU


def test_as_dict_contains_expected_sections() -> None:
    snapshot = HardwareSnapshot(
        cpu_name="Test CPU",
        cpu_logical_cores=8,
        ram_total_bytes=32_000,
        ram_available_bytes=20_000,
        cuda_available=False,
        cuda_reason="CUDA unavailable",
    )

    data = snapshot.as_dict()

    assert set(data) == {"cpu", "ram", "cuda"}
    assert data["cpu"]["name"] == "Test CPU"
    assert data["ram"]["total_bytes"] == 32_000
    assert data["cuda"]["available"] is False
    assert data["cuda"]["reason"] == "CUDA unavailable"


def test_get_posix_memory_returns_positive_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {
        "SC_PAGE_SIZE": 4096,
        "SC_PHYS_PAGES": 100,
        "SC_AVPHYS_PAGES": 40,
    }

    monkeypatch.setattr(
        hardware_monitor_module.os,
        "sysconf",
        lambda key: values[key],
        raising=False,
    )

    total, available = HardwareMonitor._get_posix_memory()

    assert total == 4096 * 100
    assert available == 4096 * 40


def test_get_posix_memory_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {
        "SC_PAGE_SIZE": 4096,
        "SC_PHYS_PAGES": 100,
        "SC_AVPHYS_PAGES": 120,
    }

    monkeypatch.setattr(
        hardware_monitor_module.os,
        "sysconf",
        lambda key: values[key],
        raising=False,
    )

    with pytest.raises(RuntimeError, match="некорректные"):
        HardwareMonitor._get_posix_memory()


def test_get_posix_memory_wraps_sysconf_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_sysconf(key: str) -> int:
        raise OSError("sysconf failed")

    monkeypatch.setattr(
        hardware_monitor_module.os,
        "sysconf",
        broken_sysconf,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="Не удалось получить"):
        HardwareMonitor._get_posix_memory()
