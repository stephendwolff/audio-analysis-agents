# JWT Authentication and Opik Project Separation

**Goal:** Protect API usage with JWT-based authentication, support demo mode with limits, and separate Opik traces by environment.

## Authentication Architecture

Three-tier auth model:

```
┌─────────────────────────────────────────────────────────┐
│                    Request Flow                          │
├─────────────────────────────────────────────────────────┤
│  1. API Key (X-API-Key header)                          │
│     └─ Environment-level gate: dev-key vs prod-key      │
│                                                          │
│  2. JWT (Authorization: Bearer <token>)                  │
│     ├─ Real users: long-lived, no request limits        │
│     └─ Demo users: 15-min expiry + 5-request limit      │
└─────────────────────────────────────────────────────────┘
```

**Page access:**
- HTML page at `/` remains public
- Actions (upload, chat) require JWT
- Users click "Try Demo" or "Login" to get a token

## Implementation

### Dependencies

Add to `pyproject.toml` api extras:
- `djangorestframework-simplejwt>=5.0.0`

### Models

Extend Django User with a profile for demo tracking:

```python
# src/api/models.py
from django.contrib.auth.models import User
from django.db import models

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    is_demo = models.BooleanField(default=False)
    request_count = models.IntegerField(default=0)
    request_limit = models.IntegerField(default=5)
```

### Auth Endpoints

| Endpoint | Input | Output |
|----------|-------|--------|
| `POST /api/auth/login/` | email, password | JWT (24h expiry) |
| `POST /api/auth/demo/` | nothing | JWT (15min expiry, is_demo=True) |
| `POST /api/auth/refresh/` | refresh token | new access token |

### Request Limiting

Middleware for demo users:
1. Decode JWT, check `is_demo` claim
2. If demo: increment `request_count` in UserProfile
3. Reject with 429 if over `request_limit`
4. Real users: no limit check

### Frontend Changes

- Add Login and "Try Demo" buttons to index.html
- Store JWT in localStorage after auth
- Attach `Authorization: Bearer <token>` to API requests
- Show remaining requests for demo users
- Prompt to login/demo when attempting actions without token

## Opik Project Separation

Update `src/observability/tracing.py`:

```python
def init_tracing(project_name: str = None) -> opik.Opik:
    project = project_name or os.getenv("OPIK_PROJECT_NAME", "audio-analysis-agents")
    # ... rest unchanged
```

Environment configuration:

| Environment | `OPIK_PROJECT_NAME` |
|-------------|---------------------|
| Local dev | `audio-analysis-agents-dev` |
| Production | `audio-analysis-agents-prod` |

Add `OPIK_PROJECT_NAME` to README Railway environment variables.

## Out of Scope

- Device UUID tracking (can add later if needed)
- Email verification / password reset
- User self-registration
