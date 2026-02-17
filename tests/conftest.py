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
                "src.api",
                "src.sketchbook",
            ],
            DEFAULT_AUTO_FIELD="django.db.models.BigAutoField",
            USE_TZ=True,
            SKETCHBOOK_MODEL_VERSION="1.0.0",
            MEDIA_ROOT="/tmp/test_media",
        )
    django.setup()


def pytest_sessionstart(session):
    """Create database tables for in-memory SQLite test DB."""
    from django.core.management import call_command

    call_command("migrate", "--run-syncdb", verbosity=0)
