"""
API URL routing.
"""

from django.urls import path

from .views import TrackUploadView, TrackDetailView, TrackListView

urlpatterns = [
    path("tracks/", TrackUploadView.as_view(), name="track-upload"),
    path("tracks/list/", TrackListView.as_view(), name="track-list"),
    path("tracks/<str:track_id>/", TrackDetailView.as_view(), name="track-detail"),
]
