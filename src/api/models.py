"""
API Models
"""

import uuid

from django.contrib.auth.models import User
from django.db import models


class Track(models.Model):
    """Audio track with cached analysis results."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4)

    # File info
    original_filename = models.CharField(max_length=255)
    storage_path = models.CharField(max_length=500)
    file_url = models.URLField(max_length=500, blank=True)
    file_size = models.PositiveIntegerField()
    duration = models.FloatField(null=True)

    # Ownership
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)

    # Status
    class Status(models.TextChoices):
        PENDING = "pending"
        ANALYZING = "analyzing"
        READY = "ready"
        FAILED = "failed"

    status = models.CharField(max_length=20, choices=Status.choices, default=Status.PENDING)
    status_message = models.TextField(blank=True)

    # Analysis results
    analysis = models.JSONField(default=dict)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    analyzed_at = models.DateTimeField(null=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.original_filename} ({self.status})"


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
