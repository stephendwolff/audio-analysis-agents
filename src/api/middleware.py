"""
API Middleware

DemoRateLimitMiddleware: Enforces request limits for demo users.
"""

from django.http import JsonResponse
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework_simplejwt.exceptions import InvalidToken

from .models import UserProfile


class DemoRateLimitMiddleware:
    """
    Middleware to enforce request limits for demo users.

    Checks JWT token for is_demo claim and increments request count.
    Returns 429 if demo user exceeds their request limit.
    """

    # Paths that don't count against limit (auth endpoints)
    EXEMPT_PATHS = [
        "/api/auth/",
        "/static/",
        "/media/",
    ]

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Skip exempt paths
        if any(request.path.startswith(path) for path in self.EXEMPT_PATHS):
            return self.get_response(request)

        # Skip if no auth header
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return self.get_response(request)

        # Try to decode JWT
        try:
            jwt_auth = JWTAuthentication()
            validated_token = jwt_auth.get_validated_token(
                auth_header.split(" ")[1]
            )
            user = jwt_auth.get_user(validated_token)
        except (InvalidToken, Exception):
            return self.get_response(request)

        # Check if demo user
        is_demo = validated_token.get("is_demo", False)
        if not is_demo:
            return self.get_response(request)

        # Get profile and check limit
        try:
            profile = user.profile
        except UserProfile.DoesNotExist:
            return self.get_response(request)

        if not profile.can_make_request():
            return JsonResponse(
                {
                    "error": "Demo request limit exceeded",
                    "request_count": profile.request_count,
                    "request_limit": profile.request_limit,
                },
                status=429,
            )

        # Increment count for API calls (not page loads)
        if request.path.startswith("/api/") and request.method in ["POST", "DELETE"]:
            profile.increment_request_count()

        return self.get_response(request)
