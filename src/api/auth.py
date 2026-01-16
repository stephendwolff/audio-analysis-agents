"""
API Key Authentication

Simple header-based API key authentication.
Set API_KEY in .env or environment.
"""

from django.conf import settings
from django.contrib.auth.models import User
from rest_framework import authentication, exceptions


class APIKeyAuthentication(authentication.BaseAuthentication):
    """
    Authenticate requests via X-API-Key header.

    Usage in views:
        authentication_classes = [APIKeyAuthentication]

    Client usage:
        curl -H "X-API-Key: your-key" http://localhost:8000/api/...
    """

    def authenticate(self, request):
        api_key = request.headers.get("X-API-Key")

        if not api_key:
            return None  # No auth attempted, let other authenticators try

        if api_key != settings.API_KEY:
            raise exceptions.AuthenticationFailed("Invalid API key")

        # Return API key user (create if needed)
        user, _ = User.objects.get_or_create(
            username="api_key_user",
            defaults={"email": "api@local", "is_active": True},
        )
        return (user, api_key)


def check_api_key(api_key: str) -> bool:
    """Check if an API key is valid. Used by WebSocket consumer."""
    return api_key == settings.API_KEY
