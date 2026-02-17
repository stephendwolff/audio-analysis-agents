"""Celery application configuration."""

import os

from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

# Setup Django before importing tasks that use models
import django
django.setup()

app = Celery("audio_analysis")
app.config_from_object("django.conf:settings", namespace="CELERY")

# Import tasks explicitly so they get registered
import src.tasks.analysis  # noqa: F401, E402
import src.sketchbook.tasks  # noqa: F401, E402
