# Railway Deployment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deploy the audio-analysis-agents Django Channels app to Railway with PostgreSQL, persistent volume storage, and configurable LLM provider.

**Architecture:** Django Channels app served via Gunicorn+Uvicorn workers, PostgreSQL for database, Railway Volume for uploaded audio files. WhiteNoise serves static files. Environment variables control all production settings.

**Tech Stack:** Django 5, Channels 4, Gunicorn, Uvicorn, PostgreSQL, WhiteNoise, dj-database-url

---

## Task 1: Add Production Dependencies to pyproject.toml

**Files:**
- Modify: `pyproject.toml:36-44` (api extras section)

**Step 1: Add whitenoise, dj-database-url, psycopg2-binary to api extras**

In `pyproject.toml`, update the `api` optional-dependencies:

```toml
api = [
    "django>=5.0.0",
    "djangorestframework>=3.14.0",
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

**Step 2: Verify dependencies install correctly**

Run: `uv pip install -e ".[api]" --dry-run`
Expected: All packages resolve without conflicts

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add production dependencies for Railway deployment"
```

---

## Task 2: Create Procfile for Railway

**Files:**
- Create: `Procfile`

**Step 1: Create Procfile with web process**

Create `Procfile` in project root:

```
web: python manage.py migrate --noinput && python manage.py collectstatic --noinput && gunicorn config.asgi:application -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT
```

**Step 2: Verify Procfile syntax**

Run: `cat Procfile`
Expected: Single line starting with `web:`

**Step 3: Commit**

```bash
git add Procfile
git commit -m "feat: add Procfile for Railway deployment"
```

---

## Task 3: Create railway.toml Configuration

**Files:**
- Create: `railway.toml`

**Step 1: Create railway.toml with nixpacks builder**

Create `railway.toml` in project root:

```toml
[build]
builder = "nixpacks"
```

**Step 2: Commit**

```bash
git add railway.toml
git commit -m "feat: add railway.toml build configuration"
```

---

## Task 4: Update Settings for Production Database

**Files:**
- Modify: `config/settings.py:1-10` (imports section)
- Modify: `config/settings.py:52-58` (database section)

**Step 1: Add dj-database-url import**

At the top of `config/settings.py`, after existing imports, add:

```python
import dj_database_url
```

Full imports section becomes:

```python
"""
Django settings for audio-analysis-agents project.
"""

import os
from pathlib import Path

import dj_database_url
from dotenv import load_dotenv

load_dotenv()
```

**Step 2: Update DATABASE configuration to use DATABASE_URL**

Replace the DATABASES section:

```python
# Database - PostgreSQL in production, SQLite for local dev
DATABASES = {
    "default": dj_database_url.config(
        default=f"sqlite:///{BASE_DIR / 'db.sqlite3'}",
        conn_max_age=600,
    )
}
```

**Step 3: Test local SQLite still works**

Run: `python manage.py check`
Expected: System check identified no issues

**Step 4: Commit**

```bash
git add config/settings.py
git commit -m "feat: add DATABASE_URL support with dj-database-url"
```

---

## Task 5: Add WhiteNoise for Static File Serving

**Files:**
- Modify: `config/settings.py:29-32` (middleware section)
- Modify: `config/settings.py:67-71` (static files section)

**Step 1: Add WhiteNoise middleware after SecurityMiddleware**

Update MIDDLEWARE in `config/settings.py`:

```python
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.middleware.common.CommonMiddleware",
]
```

**Step 2: Add STATIC_ROOT setting**

Update static files section:

```python
# Static files
STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_DIRS = [
    BASE_DIR / "static",
]
```

**Step 3: Test collectstatic works**

Run: `python manage.py collectstatic --noinput --dry-run`
Expected: Lists files that would be collected

**Step 4: Commit**

```bash
git add config/settings.py
git commit -m "feat: add WhiteNoise for production static file serving"
```

---

## Task 6: Update MEDIA_ROOT for Railway Volume

**Files:**
- Modify: `config/settings.py:73-75` (media files section)

**Step 1: Update MEDIA_ROOT to use environment variable**

Update media files section:

```python
# Media files (uploaded audio)
MEDIA_URL = "media/"
MEDIA_ROOT = Path(os.getenv("MEDIA_ROOT", BASE_DIR / "data" / "uploads"))
```

**Step 2: Test local still works**

Run: `python manage.py check`
Expected: System check identified no issues

**Step 3: Commit**

```bash
git add config/settings.py
git commit -m "feat: make MEDIA_ROOT configurable via environment"
```

---

## Task 7: Update ALLOWED_HOSTS for Railway

**Files:**
- Modify: `config/settings.py:18` (ALLOWED_HOSTS line)

**Step 1: Ensure .railway.app is supported**

The current implementation already supports this via DJANGO_ALLOWED_HOSTS env var:

```python
ALLOWED_HOSTS = os.getenv("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
```

No code change needed. This step is verification only.

**Step 2: Verify ALLOWED_HOSTS parsing works**

Run in Python shell:
```bash
python -c "import os; print('.railway.app,.example.com'.split(','))"
```
Expected: `['.railway.app', '.example.com']`

**Step 3: Commit (if any changes made)**

No commit needed if no changes.

---

## Task 8: Add staticfiles to .gitignore

**Files:**
- Modify: `.gitignore` (add staticfiles directory)

**Step 1: Add staticfiles to .gitignore**

Add to `.gitignore`:

```
staticfiles/
```

**Step 2: Commit**

```bash
git add .gitignore
git commit -m "chore: ignore staticfiles directory"
```

---

## Task 9: Final Verification

**Step 1: Run all checks**

Run: `python manage.py check --deploy`
Expected: May show warnings about HTTPS settings (expected for dev), but no errors

**Step 2: Test local server still works**

Run: `python manage.py runserver`
Expected: Server starts without errors

**Step 3: Test with production-like settings**

Run:
```bash
DJANGO_DEBUG=false DJANGO_SECRET_KEY=test-key python manage.py check
```
Expected: No errors

---

## Railway Setup Reference (Manual Steps After Deployment)

These are done in Railway dashboard, not code:

### 1. Create PostgreSQL Service
- Railway dashboard → New → Database → PostgreSQL
- Automatically provides DATABASE_URL

### 2. Create Volume for Media
- Service → Settings → Volumes
- Mount path: `/app/data/uploads`
- Set env var: `MEDIA_ROOT=/app/data/uploads`

### 3. Set Environment Variables
```
DJANGO_SECRET_KEY=<generate-secure-key>
DJANGO_DEBUG=false
DJANGO_ALLOWED_HOSTS=.railway.app
API_KEY=<your-api-key>
LLM_MODEL=gemini/gemini-2.0-flash
GOOGLE_API_KEY=<key>
ANTHROPIC_API_KEY=<key>
MEDIA_ROOT=/app/data/uploads
```

### 4. Connect GitHub Repository
- Railway dashboard → New Project → Deploy from GitHub
- Select repository

### 5. Verification Endpoints
- `GET /` - returns HTML
- `POST /api/tracks/` - upload works
- `GET /api/tracks/<id>/` - retrieval works
- `ws://host/ws/chat/<id>/` - WebSocket works
