"""
WebSocket URL routing.
"""

from django.urls import path

from src.api.consumers import ChatConsumer

websocket_urlpatterns = [
    path("ws/chat/<str:track_id>/", ChatConsumer.as_asgi()),
]
