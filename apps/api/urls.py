from django.urls import path
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from services.probability import full_combination_probability

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def health(request):
    return Response({"status": "ok", "service": "LAIE", "version": "1.0.0"})

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def baseline(request):
    return Response({
        "game": "rapido-pro",
        "format": "8/20+1/4",
        "full_match_probability": full_combination_probability(),
        "note": "Теоретическая вероятность при равновероятном розыгрыше; не прогноз.",
    })

urlpatterns = [path("health/", health), path("baseline/", baseline)]
