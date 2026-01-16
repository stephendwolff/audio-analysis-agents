# JWT Authentication Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add JWT-based authentication with demo mode (15-min / 5-request limit) and separate Opik projects by environment.

**Architecture:** Django User model + UserProfile for demo tracking. simplejwt for token handling. Middleware checks demo limits. Frontend stores JWT in localStorage.

**Tech Stack:** Django, djangorestframework-simplejwt, existing DRF setup

---

## Task 1: Add JWT Dependency

**Files:**
- Modify: `pyproject.toml:36-47` (api extras section)

**Step 1: Add djangorestframework-simplejwt to api extras**

In `pyproject.toml`, add to the `api` optional-dependencies:

```toml
api = [
    "django>=5.0.0",
    "djangorestframework>=3.14.0",
    "djangorestframework-simplejwt>=5.0.0",
    "channels>=4.0.0",
    "uvicorn[standard]>=0.30.0",
    "gunicorn>=22.0.0",
    "python-dotenv>=1.0.0",
    "google-genai>=1.0.0",
    "whitenoise>=6.0.0",
    "dj-database-url>=2.0.0",
    "psycopg2-binary>=2.9.0",
]
```

**Step 2: Install dependencies**

Run: `uv pip install -e ".[api]"`
Expected: Successfully installed djangorestframework-simplejwt

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add djangorestframework-simplejwt dependency"
```

---

## Task 2: Create UserProfile Model

**Files:**
- Modify: `src/api/models.py` (create if doesn't exist)

**Step 1: Create models.py with UserProfile**

Create `src/api/models.py`:

```python
"""
API Models
"""

from django.contrib.auth.models import User
from django.db import models


class UserProfile(models.Model):
    """
    Extended user profile for tracking demo usage.
    """

    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="profile")
    is_demo = models.BooleanField(default=False)
    request_count = models.IntegerField(default=0)
    request_limit = models.IntegerField(default=5)

    def __str__(self):
        return f"{self.user.username} ({'demo' if self.is_demo else 'user'})"

    def can_make_request(self) -> bool:
        """Check if user can make another request."""
        if not self.is_demo:
            return True
        return self.request_count < self.request_limit

    def increment_request_count(self) -> None:
        """Increment request count for demo users."""
        if self.is_demo:
            self.request_count += 1
            self.save(update_fields=["request_count"])
```

**Step 2: Create and run migrations**

Run: `python manage.py makemigrations api && python manage.py migrate`
Expected: Migrations created and applied

**Step 3: Commit**

```bash
git add src/api/models.py src/api/migrations/
git commit -m "feat: add UserProfile model for demo tracking"
```

---

## Task 3: Configure JWT Settings

**Files:**
- Modify: `config/settings.py`

**Step 1: Add simplejwt to INSTALLED_APPS**

In `config/settings.py`, update INSTALLED_APPS:

```python
INSTALLED_APPS = [
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.staticfiles",
    "rest_framework",
    "rest_framework_simplejwt",
    "channels",
    "src.api",
]
```

**Step 2: Add JWT configuration**

Add after REST_FRAMEWORK section:

```python
# JWT Configuration
from datetime import timedelta

SIMPLE_JWT = {
    "ACCESS_TOKEN_LIFETIME": timedelta(hours=24),
    "REFRESH_TOKEN_LIFETIME": timedelta(days=7),
    "AUTH_HEADER_TYPES": ("Bearer",),
    "USER_ID_FIELD": "id",
    "USER_ID_CLAIM": "user_id",
}

# Demo token lifetime (shorter)
DEMO_TOKEN_LIFETIME = timedelta(minutes=15)
DEMO_REQUEST_LIMIT = 5
```

**Step 3: Update REST_FRAMEWORK settings**

Update REST_FRAMEWORK to include JWT authentication:

```python
REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework_simplejwt.authentication.JWTAuthentication",
        "src.api.auth.APIKeyAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": [],
    "UNAUTHENTICATED_USER": None,
}
```

**Step 4: Verify settings load**

Run: `python manage.py check`
Expected: System check identified no issues

**Step 5: Commit**

```bash
git add config/settings.py
git commit -m "feat: configure JWT authentication settings"
```

---

## Task 4: Create Auth Views

**Files:**
- Create: `src/api/auth_views.py`
- Modify: `src/api/urls.py`

**Step 1: Create auth_views.py**

Create `src/api/auth_views.py`:

```python
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
```

**Step 2: Add auth URLs**

Update `src/api/urls.py`:

```python
"""
API URL routing.
"""

from django.urls import path

from .auth_views import LoginView, DemoView, RefreshView
from .views import TrackUploadView, TrackDetailView, TrackListView

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
```

**Step 3: Run migrations for auth app**

Run: `python manage.py migrate`
Expected: Auth tables created (if not already)

**Step 4: Verify endpoints exist**

Run: `python manage.py show_urls 2>/dev/null | grep auth || echo "URLs registered"`
Expected: Auth URLs listed or confirmation

**Step 5: Commit**

```bash
git add src/api/auth_views.py src/api/urls.py
git commit -m "feat: add JWT auth endpoints (login, demo, refresh)"
```

---

## Task 5: Create Demo Rate Limit Middleware

**Files:**
- Create: `src/api/middleware.py`
- Modify: `config/settings.py`

**Step 1: Create middleware.py**

Create `src/api/middleware.py`:

```python
"""
API Middleware

DemoRateLimitMiddleware: Enforces request limits for demo users.
"""

import json

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
```

**Step 2: Add middleware to settings**

In `config/settings.py`, update MIDDLEWARE:

```python
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.middleware.common.CommonMiddleware",
    "src.api.middleware.DemoRateLimitMiddleware",
]
```

**Step 3: Verify middleware loads**

Run: `python manage.py check`
Expected: System check identified no issues

**Step 4: Commit**

```bash
git add src/api/middleware.py config/settings.py
git commit -m "feat: add demo rate limit middleware"
```

---

## Task 6: Update Views to Use JWT

**Files:**
- Modify: `src/api/views.py`
- Modify: `src/api/auth.py`

**Step 1: Update views.py authentication**

In `src/api/views.py`, update imports and authentication:

```python
"""
API Views for audio upload and track management.
"""

import uuid
from pathlib import Path

from django.conf import settings
from rest_framework import status
from rest_framework.parsers import MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.authentication import JWTAuthentication

from .auth import APIKeyAuthentication


# In-memory track storage (for demo - use database in production)
TRACKS: dict[str, dict] = {}


class TrackUploadView(APIView):
    """
    Upload an audio file.

    POST /api/tracks/
    Headers: Authorization: Bearer <token>
    Body: multipart/form-data with 'file' field
    """

    authentication_classes = [JWTAuthentication, APIKeyAuthentication]
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser]

    def post(self, request):
        # ... rest unchanged
```

Update `TrackDetailView` and `TrackListView` similarly:

```python
class TrackDetailView(APIView):
    """Get or delete a track."""

    authentication_classes = [JWTAuthentication, APIKeyAuthentication]
    permission_classes = [IsAuthenticated]

    # ... methods unchanged


class TrackListView(APIView):
    """List all tracks."""

    authentication_classes = [JWTAuthentication, APIKeyAuthentication]
    permission_classes = [IsAuthenticated]

    # ... methods unchanged
```

**Step 2: Update APIKeyAuthentication to return a user**

Update `src/api/auth.py` to create/return an anonymous user:

```python
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
```

**Step 3: Run migrations**

Run: `python manage.py migrate`
Expected: No new migrations needed

**Step 4: Test endpoints still work**

Run: `python manage.py check`
Expected: System check identified no issues

**Step 5: Commit**

```bash
git add src/api/views.py src/api/auth.py
git commit -m "feat: update views to support JWT authentication"
```

---

## Task 7: Update WebSocket Consumer for JWT

**Files:**
- Modify: `src/api/consumers.py`

**Step 1: Update consumer to accept JWT**

Update `src/api/consumers.py`:

```python
"""
WebSocket Consumer for Streaming Chat
"""

import json
import os
from urllib.parse import parse_qs

from channels.generic.websocket import AsyncWebsocketConsumer
from django.conf import settings
from rest_framework_simplejwt.tokens import AccessToken
from rest_framework_simplejwt.exceptions import InvalidToken, TokenError

from .auth import check_api_key
from .views import get_track_path


class ChatConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer for streaming audio analysis chat.

    Accepts either:
    - ?token=<jwt> - JWT authentication
    - ?api_key=<key> - Legacy API key authentication
    """

    async def connect(self):
        """Handle WebSocket connection."""
        self.track_id = self.scope["url_route"]["kwargs"]["track_id"]
        self.user = None
        self.is_demo = False

        # Parse query string
        query_string = self.scope.get("query_string", b"").decode()
        params = parse_qs(query_string)

        # Try JWT auth first
        token = params.get("token", [None])[0]
        if token:
            try:
                access_token = AccessToken(token)
                self.is_demo = access_token.get("is_demo", False)
                # JWT is valid
            except (InvalidToken, TokenError):
                await self.close(code=4001)  # Unauthorized
                return
        else:
            # Fall back to API key
            api_key = params.get("api_key", [None])[0]
            if not api_key or not check_api_key(api_key):
                await self.close(code=4001)  # Unauthorized
                return

        # Check track exists
        self.track_path = get_track_path(self.track_id)
        if not self.track_path:
            await self.close(code=4004)  # Not found
            return

        await self.accept()

    # ... rest of methods unchanged
```

**Step 2: Verify server starts**

Run: `python manage.py check`
Expected: System check identified no issues

**Step 3: Commit**

```bash
git add src/api/consumers.py
git commit -m "feat: add JWT support to WebSocket consumer"
```

---

## Task 8: Update Frontend for JWT Auth

**Files:**
- Modify: `templates/index.html`
- Modify: `static/js/app.js`
- Modify: `static/js/config.js`

**Step 1: Update index.html with auth section**

Replace the config-section in `templates/index.html`:

```html
{% load static %}
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Audio Analysis - Example Frontend</title>
  <link rel="stylesheet" href="{% static 'css/style.css' %}">
</head>
<body>
  <h1>Audio Analysis</h1>
  <p>Upload an audio file and ask questions about it using natural language.</p>

  <!-- Auth Section -->
  <section id="auth-section">
    <h2>Authentication</h2>
    <div id="auth-status"></div>
    <div id="auth-buttons">
      <button id="demo-btn">Try Demo (5 requests)</button>
      <button id="login-btn">Login</button>
    </div>
    <form id="login-form" style="display: none;">
      <div class="form-group">
        <label for="email">Email</label>
        <input type="email" id="email" placeholder="user@example.com">
      </div>
      <div class="form-group">
        <label for="password">Password</label>
        <input type="password" id="password" placeholder="password">
      </div>
      <button type="submit">Login</button>
      <button type="button" id="cancel-login">Cancel</button>
    </form>
  </section>

  <!-- Upload -->
  <section id="upload-section" style="display: none;">
    <h2>Upload Audio</h2>
    <form id="upload-form">
      <div class="form-group">
        <label for="file-input">Audio File</label>
        <input type="file" id="file-input" accept=".wav,.mp3,.flac,.ogg,.m4a">
      </div>
      <button type="submit">Upload</button>
      <div id="upload-status" class="status"></div>
    </form>
  </section>

  <!-- Chat -->
  <section id="chat-section" style="display: none;">
    <h2>Ask Questions <span id="connection-status" class="status"></span></h2>
    <div id="track-info"></div>
    <div id="demo-info"></div>
    <div id="chat-messages"></div>
    <form id="chat-form">
      <input type="text" id="question-input" placeholder="What's the tempo of this track?" disabled>
      <button type="submit">Send</button>
    </form>
  </section>

  <!-- Scripts -->
  <script src="{% static 'js/config.js' %}"></script>
  <script src="{% static 'js/app.js' %}"></script>
</body>
</html>
```

**Step 2: Update config.js**

Update `static/js/config.js`:

```javascript
/**
 * Configuration for the Audio Analysis Frontend
 */

const CONFIG = {
  // API base URL (no trailing slash)
  API_BASE_URL: window.location.origin,

  // WebSocket base URL (ws:// or wss://)
  WS_BASE_URL: window.location.origin.replace(/^http/, "ws"),

  // JWT token (set after login/demo)
  ACCESS_TOKEN: null,
  REFRESH_TOKEN: null,
  IS_DEMO: false,
  REQUEST_LIMIT: 0,
};

// Make config available globally
window.CONFIG = CONFIG;
```

**Step 3: Update app.js with auth logic**

Replace `static/js/app.js`:

```javascript
/**
 * Audio Analysis Frontend
 */

// State
let currentTrackId = null;
let websocket = null;
let requestCount = 0;

// DOM Elements
const authSection = document.getElementById("auth-section");
const authStatus = document.getElementById("auth-status");
const authButtons = document.getElementById("auth-buttons");
const demoBtn = document.getElementById("demo-btn");
const loginBtn = document.getElementById("login-btn");
const loginForm = document.getElementById("login-form");
const cancelLoginBtn = document.getElementById("cancel-login");
const uploadSection = document.getElementById("upload-section");
const uploadForm = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const uploadStatus = document.getElementById("upload-status");
const chatSection = document.getElementById("chat-section");
const trackInfo = document.getElementById("track-info");
const demoInfo = document.getElementById("demo-info");
const chatMessages = document.getElementById("chat-messages");
const chatForm = document.getElementById("chat-form");
const questionInput = document.getElementById("question-input");
const connectionStatus = document.getElementById("connection-status");

// Auth handlers
demoBtn.addEventListener("click", async () => {
  try {
    const response = await fetch(`${CONFIG.API_BASE_URL}/api/auth/demo/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });

    if (!response.ok) throw new Error("Failed to start demo");

    const data = await response.json();
    CONFIG.ACCESS_TOKEN = data.access;
    CONFIG.REFRESH_TOKEN = data.refresh;
    CONFIG.IS_DEMO = true;
    CONFIG.REQUEST_LIMIT = data.request_limit;
    requestCount = 0;

    showAuthenticatedUI();
  } catch (error) {
    authStatus.textContent = `Error: ${error.message}`;
    authStatus.className = "status error";
  }
});

loginBtn.addEventListener("click", () => {
  authButtons.style.display = "none";
  loginForm.style.display = "block";
});

cancelLoginBtn.addEventListener("click", () => {
  loginForm.style.display = "none";
  authButtons.style.display = "block";
});

loginForm.addEventListener("submit", async (e) => {
  e.preventDefault();

  const email = document.getElementById("email").value;
  const password = document.getElementById("password").value;

  try {
    const response = await fetch(`${CONFIG.API_BASE_URL}/api/auth/login/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || "Login failed");
    }

    const data = await response.json();
    CONFIG.ACCESS_TOKEN = data.access;
    CONFIG.REFRESH_TOKEN = data.refresh;
    CONFIG.IS_DEMO = data.is_demo;

    showAuthenticatedUI();
  } catch (error) {
    authStatus.textContent = `Error: ${error.message}`;
    authStatus.className = "status error";
  }
});

function showAuthenticatedUI() {
  authButtons.style.display = "none";
  loginForm.style.display = "none";

  if (CONFIG.IS_DEMO) {
    authStatus.textContent = `Demo mode: ${CONFIG.REQUEST_LIMIT} requests available`;
  } else {
    authStatus.textContent = "Logged in";
  }
  authStatus.className = "status success";

  uploadSection.style.display = "block";
}

// Upload file
uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();

  const file = fileInput.files[0];
  if (!file) {
    showUploadStatus("Please select a file", "error");
    return;
  }

  showUploadStatus("Uploading...", "info");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch(`${CONFIG.API_BASE_URL}/api/tracks/`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${CONFIG.ACCESS_TOKEN}`,
      },
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || "Upload failed");
    }

    const data = await response.json();
    currentTrackId = data.track_id;

    showUploadStatus(`Uploaded: ${data.filename}`, "success");
    showChatSection(data);
  } catch (error) {
    showUploadStatus(`Error: ${error.message}`, "error");
  }
});

// Show chat section and connect WebSocket
function showChatSection(trackData) {
  chatSection.style.display = "block";
  trackInfo.textContent = `Track: ${trackData.filename} (${trackData.track_id})`;
  chatMessages.innerHTML = "";
  updateDemoInfo();

  connectWebSocket(trackData.track_id);
}

function updateDemoInfo() {
  if (CONFIG.IS_DEMO) {
    const remaining = CONFIG.REQUEST_LIMIT - requestCount;
    demoInfo.textContent = `Demo: ${remaining} requests remaining`;
    demoInfo.className = remaining <= 1 ? "status error" : "status info";
  } else {
    demoInfo.textContent = "";
  }
}

// Connect to WebSocket
function connectWebSocket(trackId) {
  if (websocket) {
    websocket.close();
  }

  const wsUrl = `${CONFIG.WS_BASE_URL}/ws/chat/${trackId}/?token=${encodeURIComponent(CONFIG.ACCESS_TOKEN)}`;
  setConnectionStatus("Connecting...", "info");

  websocket = new WebSocket(wsUrl);

  websocket.onopen = () => {
    setConnectionStatus("Connected", "success");
    questionInput.disabled = false;
  };

  websocket.onclose = (e) => {
    if (e.code === 4001) {
      setConnectionStatus("Unauthorized", "error");
    } else if (e.code === 4004) {
      setConnectionStatus("Track not found", "error");
    } else {
      setConnectionStatus("Disconnected", "error");
    }
    questionInput.disabled = true;
  };

  websocket.onerror = () => {
    setConnectionStatus("Connection error", "error");
  };

  websocket.onmessage = (event) => {
    handleWebSocketMessage(JSON.parse(event.data));
  };
}

// Handle incoming WebSocket messages
let currentResponseElement = null;

function handleWebSocketMessage(data) {
  switch (data.type) {
    case "tool_call":
      addSystemMessage(`Calling tool: ${data.tool}`);
      break;

    case "tool_result":
      const status = data.success ? "✓" : "✗";
      addSystemMessage(`Tool ${data.tool}: ${status}`);
      break;

    case "token":
      if (!currentResponseElement) {
        currentResponseElement = addAssistantMessage("");
      }
      currentResponseElement.textContent += data.text;
      scrollToBottom();
      break;

    case "done":
      currentResponseElement = null;
      break;

    case "error":
      addErrorMessage(data.message);
      currentResponseElement = null;
      break;
  }
}

// Send question
chatForm.addEventListener("submit", (e) => {
  e.preventDefault();

  const question = questionInput.value.trim();
  if (!question || !websocket || websocket.readyState !== WebSocket.OPEN) {
    return;
  }

  // Check demo limit
  if (CONFIG.IS_DEMO && requestCount >= CONFIG.REQUEST_LIMIT) {
    addErrorMessage("Demo request limit reached. Please login for unlimited access.");
    return;
  }

  addUserMessage(question);
  questionInput.value = "";

  if (CONFIG.IS_DEMO) {
    requestCount++;
    updateDemoInfo();
  }

  websocket.send(
    JSON.stringify({
      type: "question",
      text: question,
    })
  );
});

// UI Helpers
function showUploadStatus(message, type) {
  uploadStatus.textContent = message;
  uploadStatus.className = `status ${type}`;
}

function setConnectionStatus(message, type) {
  connectionStatus.textContent = message;
  connectionStatus.className = `status ${type}`;
}

function addUserMessage(text) {
  const div = document.createElement("div");
  div.className = "message user";
  div.textContent = text;
  chatMessages.appendChild(div);
  scrollToBottom();
}

function addAssistantMessage(text) {
  const div = document.createElement("div");
  div.className = "message assistant";
  div.textContent = text;
  chatMessages.appendChild(div);
  scrollToBottom();
  return div;
}

function addSystemMessage(text) {
  const div = document.createElement("div");
  div.className = "message system";
  div.textContent = text;
  chatMessages.appendChild(div);
  scrollToBottom();
}

function addErrorMessage(text) {
  const div = document.createElement("div");
  div.className = "message error";
  div.textContent = `Error: ${text}`;
  chatMessages.appendChild(div);
  scrollToBottom();
}

function scrollToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}
```

**Step 4: Commit**

```bash
git add templates/index.html static/js/app.js static/js/config.js
git commit -m "feat: update frontend for JWT authentication"
```

---

## Task 9: Update Opik Tracing

**Files:**
- Modify: `src/observability/tracing.py`

**Step 1: Update init_tracing to use env var**

Update `src/observability/tracing.py`:

```python
def init_tracing(project_name: str = None) -> opik.Opik:
    """
    Initialise Opik tracing.

    Call this once at application startup.

    Args:
        project_name: Name of the project in Opik dashboard.
                      Defaults to OPIK_PROJECT_NAME env var or "audio-analysis-agents".

    Returns:
        Opik client instance
    """
    global _client

    api_key = os.getenv("OPIK_API_KEY")
    if not api_key:
        print("Warning: OPIK_API_KEY not set, tracing disabled")
        return None

    project = project_name or os.getenv("OPIK_PROJECT_NAME", "audio-analysis-agents")
    _client = opik.Opik(project_name=project)
    print(f"Opik tracing initialised for project: {project}")
    return _client
```

**Step 2: Verify it loads**

Run: `python -c "from src.observability.tracing import init_tracing; print('OK')"`
Expected: OK

**Step 3: Commit**

```bash
git add src/observability/tracing.py
git commit -m "feat: make Opik project name configurable via env var"
```

---

## Task 10: Final Verification

**Step 1: Run all checks**

Run: `python manage.py check`
Expected: System check identified no issues

**Step 2: Run migrations**

Run: `python manage.py migrate`
Expected: All migrations applied

**Step 3: Create a test user**

Run:
```bash
python manage.py shell -c "
from django.contrib.auth.models import User
from src.api.models import UserProfile
user, created = User.objects.get_or_create(username='testuser', defaults={'email': 'test@example.com'})
if created:
    user.set_password('testpass123')
    user.save()
    UserProfile.objects.create(user=user)
    print('Created test user: test@example.com / testpass123')
else:
    print('Test user already exists')
"
```

**Step 4: Test server starts**

Run: `timeout 3 python manage.py runserver 8001 2>&1 || true`
Expected: Server starts without errors

**Step 5: Run tests**

Run: `pytest`
Expected: All tests pass

**Step 6: Commit any remaining changes**

```bash
git status
# If any uncommitted changes, commit them
```
