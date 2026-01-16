"""
Authentication Views

Endpoints:
- POST /api/auth/login/ - Login with email/password, get JWT
- POST /api/auth/demo/ - Get demo JWT (limited usage)
- POST /api/auth/refresh/ - Refresh access token
"""

import uuid

from django.conf import settings
from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import RefreshToken

from .models import UserProfile


class LoginView(APIView):
    """
    Login with email/password to get JWT tokens.

    POST /api/auth/login/
    Body: {"email": "user@example.com", "password": "secret"}

    Returns:
        {"access": "...", "refresh": "...", "is_demo": false}
    """

    authentication_classes = []  # Public endpoint

    def post(self, request):
        email = request.data.get("email")
        password = request.data.get("password")

        if not email or not password:
            return Response(
                {"error": "Email and password required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Django uses username for auth, but we accept email
        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            return Response(
                {"error": "Invalid credentials"},
                status=status.HTTP_401_UNAUTHORIZED,
            )

        user = authenticate(username=user.username, password=password)
        if not user:
            return Response(
                {"error": "Invalid credentials"},
                status=status.HTTP_401_UNAUTHORIZED,
            )

        # Get or create profile
        profile, _ = UserProfile.objects.get_or_create(user=user)

        refresh = RefreshToken.for_user(user)
        refresh["is_demo"] = profile.is_demo

        return Response({
            "access": str(refresh.access_token),
            "refresh": str(refresh),
            "is_demo": profile.is_demo,
        })


class DemoView(APIView):
    """
    Get a demo JWT with limited usage.

    POST /api/auth/demo/
    Body: {} (empty)

    Returns:
        {"access": "...", "refresh": "...", "is_demo": true, "request_limit": 5}
    """

    authentication_classes = []  # Public endpoint

    def post(self, request):
        # Create temporary demo user
        demo_username = f"demo_{uuid.uuid4().hex[:8]}"
        demo_user = User.objects.create_user(
            username=demo_username,
            email=f"{demo_username}@demo.local",
            password=uuid.uuid4().hex,
        )

        # Create demo profile
        profile = UserProfile.objects.create(
            user=demo_user,
            is_demo=True,
            request_limit=settings.DEMO_REQUEST_LIMIT,
        )

        # Create tokens with shorter lifetime for demo
        refresh = RefreshToken.for_user(demo_user)
        refresh["is_demo"] = True

        # Override access token lifetime for demo
        refresh.access_token.set_exp(lifetime=settings.DEMO_TOKEN_LIFETIME)

        return Response({
            "access": str(refresh.access_token),
            "refresh": str(refresh),
            "is_demo": True,
            "request_limit": profile.request_limit,
        })


class RefreshView(APIView):
    """
    Refresh access token.

    POST /api/auth/refresh/
    Body: {"refresh": "..."}

    Returns:
        {"access": "..."}
    """

    authentication_classes = []  # Public endpoint

    def post(self, request):
        refresh_token = request.data.get("refresh")

        if not refresh_token:
            return Response(
                {"error": "Refresh token required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            refresh = RefreshToken(refresh_token)
            return Response({
                "access": str(refresh.access_token),
            })
        except Exception:
            return Response(
                {"error": "Invalid refresh token"},
                status=status.HTTP_401_UNAUTHORIZED,
            )
