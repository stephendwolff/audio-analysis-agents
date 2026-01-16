"""
API URL routing.
"""

from django.urls import path

from .auth_views import DemoView, LoginView, RefreshView
from .views import TrackDetailView, TrackListView, TrackUploadView

urlpatterns = [
    # Auth endpoints
    path("auth/login/", LoginView.as_view(), name="auth-login"),
    path("auth/demo/", DemoView.as_view(), name="auth-demo"),
    path("auth/refresh/", RefreshView.as_view(), name="auth-refresh"),
    # Track endpoints
    path("tracks/", TrackUploadView.as_view(), name="track-upload"),
    path("tracks/list/", TrackListView.as_view(), name="track-list"),
    path("tracks/<str:track_id>/", TrackDetailView.as_view(), name="track-detail"),
]
