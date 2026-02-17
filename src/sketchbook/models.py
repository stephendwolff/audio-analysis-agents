"""Models for the Musical Sketchbook API."""

from django.contrib.auth.models import User
from django.db import models


class Fragment(models.Model):
    """An audio fragment submitted by the Musical Sketchbook iOS app."""

    fragment_id = models.UUIDField(unique=True, db_index=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="fragments")

    audio_storage_path = models.CharField(max_length=500)
    mime_type = models.CharField(max_length=100)
    duration_seconds = models.FloatField()

    class Status(models.TextChoices):
        PENDING = "pending"
        ANALYZING = "analyzing"
        COMPLETE = "complete"
        FAILED = "failed"

    status = models.CharField(max_length=20, choices=Status, default=Status.PENDING)
    error_message = models.TextField(blank=True)

    model_version = models.CharField(max_length=20)
    analysis = models.JSONField(default=dict)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Fragment {self.fragment_id} ({self.status})"
