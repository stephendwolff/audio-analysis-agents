"""URL routing for the Musical Sketchbook API."""

from django.urls import path
from . import views

urlpatterns = [
    path("health/", views.HealthView.as_view(), name="sketchbook-health"),
    path("analyse/", views.AnalyseView.as_view(), name="sketchbook-analyse"),
    path("analyse/<uuid:fragment_id>/", views.AnalyseView.as_view(), name="sketchbook-analyse-detail"),
]
