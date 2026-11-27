from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.http import require_GET

from services.compute.engine import ComputeEngine


@require_GET
def status(request):
    """Диагностика локального вычислительного оборудования."""
    engine = ComputeEngine(
        min_gpu_items=settings.LAIE_COMPUTE_MIN_GPU_ITEMS,
        memory_reserve_mb=settings.LAIE_GPU_MEMORY_RESERVE_MB,
        memory_reserve_ratio=settings.LAIE_GPU_MEMORY_RESERVE_RATIO,
    )
    return JsonResponse(engine.inspect())
