from django.urls import path

from . import views

app_name = "compute"
urlpatterns = [path("status/", views.status, name="status")]
