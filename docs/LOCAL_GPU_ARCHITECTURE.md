# LAIE Local Autonomous Edition — GPU architecture

## Mandatory architecture rules

- Django binds only to `127.0.0.1`; do not bind to `0.0.0.0` or a LAN address.
- PostgreSQL, source data, experiment artifacts, model checkpoints and audit data remain local.
- No cloud workers, cloud jobs, remote compute endpoints, external AI APIs or remote database access.
- Internet access is not required at runtime after installation and local data acquisition.
- Ticket purchase is always a separate manual user action.

## NVIDIA compute

`services.compute.engine.ComputeEngine` probes PyTorch lazily. If PyTorch/CUDA is
missing or the driver cannot be queried, the application remains operational on CPU.
For large tasks, the engine can select CUDA when the estimated workload crosses
`LAIE_COMPUTE_MIN_GPU_ITEMS` and enough free VRAM remains after the configured reserve.
If a CUDA tensor operation fails, the sample operation retries on CPU.

The current hardware's exact GPU model and driver version must be checked before
choosing a PyTorch CUDA wheel. Use the official Windows/Pip selector at:
https://pytorch.org/get-started/locally/
Select Windows, Pip, Python and a CUDA build offered by the selector. Do not install
an arbitrary CUDA Toolkit version just because its number is newer. The PyTorch wheel
contains the CUDA runtime components it needs; the NVIDIA display driver must still
be compatible. NVIDIA's Windows CUDA guide describes supported drivers and GPUs:
https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/

## Runtime endpoint

`GET /api/v1/compute/status/` reports PyTorch presence, CUDA availability, GPU model,
and VRAM information when the installed driver exposes it.

## Configuration

- `LAIE_COMPUTE_MIN_GPU_ITEMS`: workload threshold for automatic CUDA selection.
- `LAIE_GPU_MEMORY_RESERVE_MB`: VRAM safety reserve.

The thresholds are initial conservative defaults and must be tuned by local benchmark
results. Small statistical jobs should remain on CPU because host/device transfers can
cost more than the GPU computation itself.
