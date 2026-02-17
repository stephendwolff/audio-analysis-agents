import django
from django.conf import settings


def pytest_configure():
    """Configure Django settings before tests."""
    if not settings.configured:
        settings.configure(
            DEBUG=True,
            SECRET_KEY="test-secret-key-for-testing-only",
            DATABASES={
                "default": {
                    "ENGINE": "django.db.backends.sqlite3",
                    "NAME": ":memory:",
                }
            },
            INSTALLED_APPS=[
                "django.contrib.auth",
                "django.contrib.contenttypes",
                "rest_framework",
                "src.api",
                "src.sketchbook",
            ],
            DEFAULT_AUTO_FIELD="django.db.models.BigAutoField",
            USE_TZ=True,
            ROOT_URLCONF="config.urls",
            REST_FRAMEWORK={
                "DEFAULT_AUTHENTICATION_CLASSES": [
                    "rest_framework_simplejwt.authentication.JWTAuthentication",
                ],
                "DEFAULT_PERMISSION_CLASSES": [],
                "UNAUTHENTICATED_USER": None,
            },
            SKETCHBOOK_MODEL_VERSION="1.0.0",
            MEDIA_ROOT="/tmp/test_media",
            MEDIA_URL="media/",
            STATIC_URL="static/",
            STATICFILES_DIRS=["/tmp/test_static"],
            TEMPLATES=[{
                "BACKEND": "django.template.backends.django.DjangoTemplates",
                "DIRS": [],
                "APP_DIRS": True,
                "OPTIONS": {"context_processors": []},
            }],
        )
    django.setup()


def pytest_sessionstart(session):
    """Create database tables for in-memory SQLite test DB."""
    from django.core.management import call_command

    call_command("migrate", "--run-syncdb", verbosity=0)
