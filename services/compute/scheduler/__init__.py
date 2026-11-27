
"""Публичный интерфейс планировщика вычислений LAIE."""

from services.compute.scheduler.task_registry import (
    TaskAlreadyRegisteredError,
    TaskNotFoundError,
    TaskRegistry,
    create_default_registry,
)
from services.compute.scheduler.task_types import (
    ComputeBackend,
    PrecisionRequirement,
    TaskDefinition,
    TaskType,
    WorkloadKind,
)

from services.compute.scheduler.task_analyzer import (
    TaskAnalysis,
    TaskAnalyzer,
    TaskParameters,
)

from services.compute.scheduler.performance_model import (
    PerformanceEstimate,
    PerformanceModel,
    PerformanceSample,
    UnsupportedBackendError,
)

from .memory_estimator import (
    MemoryDeviceSnapshot,
    MemoryEstimate,
    MemoryEstimator,
)

from .decision_engine import (
    DecisionEngine,
    DecisionReason,
    DecisionResult,
    DecisionStatus,
)

from .hardware_monitor import HardwareMonitor, HardwareSnapshot

from .pytorch_adapter import (
    BackendUnavailableError,
    CUDAExecutionError,
    InvalidTensorError,
    MatmulExecution,
    PyTorchCUDAAdapter,
    PyTorchAdapterError,
)

__all__ = [
    "ComputeBackend",
    "PrecisionRequirement",
    "TaskAlreadyRegisteredError",
    "TaskDefinition",
    "TaskNotFoundError",
    "TaskRegistry",
    "TaskType",
    "WorkloadKind",
    "create_default_registry",
    "TaskAnalysis",
    "TaskAnalyzer",
    "TaskParameters",
    "PerformanceEstimate",
    "PerformanceModel",
    "PerformanceSample",
    "UnsupportedBackendError",
    "MemoryDeviceSnapshot",
    "MemoryEstimate",
    "MemoryEstimator",
    "DecisionEngine",
    "DecisionReason",
    "DecisionResult",
    "DecisionStatus",
    "HardwareMonitor",
    "HardwareSnapshot",
    "BackendUnavailableError",
    "CUDAExecutionError",
    "InvalidTensorError",
    "MatmulExecution",
    "PyTorchCUDAAdapter",
    "PyTorchAdapterError",

]
