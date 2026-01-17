"""Celery tasks for background audio analysis."""

from .celery import app as celery_app

__all__ = ["celery_app"]
