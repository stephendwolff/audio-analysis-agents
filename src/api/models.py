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

    # ----- Class methods -----

    @classmethod
    def get_or_none(cls, track_id: str) -> "Track | None":
        """Get track by ID, returning None if not found."""
        try:
            return cls.objects.get(id=track_id)
        except cls.DoesNotExist:
            return None

    # ----- Properties -----

    @property
    def file_path(self) -> str:
        """Resolve storage path to filesystem path or URL."""
        from django.core.files.storage import default_storage
        if hasattr(default_storage, "path"):
            try:
                return default_storage.path(self.storage_path)
            except NotImplementedError:
                pass
        return self.file_url

    @property
    def is_ready(self) -> bool:
        """Check if track is ready for questions."""
        return self.status == self.Status.READY

    # ----- Analysis methods -----

    def has_analysis(self, agent_name: str) -> bool:
        """Check if analysis exists and has no error."""
        if agent_name not in self.analysis:
            return False
        data = self.analysis[agent_name]
        return not (isinstance(data, dict) and "error" in data)

    def get_analysis(self, agent_name: str) -> dict:
        """Get analysis result. Raises KeyError/ValueError if unavailable."""
        if agent_name not in self.analysis:
            raise KeyError(f"No {agent_name} analysis for track {self.id}")
        data = self.analysis[agent_name]
        if isinstance(data, dict) and "error" in data:
            raise ValueError(data["error"])
        return data

    def set_analysis(self, agent_name: str, data: dict) -> None:
        """Store analysis result."""
        self.analysis[agent_name] = data
        self.save(update_fields=["analysis"])

    def queue_analysis(self) -> str | None:
        """Queue background analysis task. Returns task_id or None on failure."""
        from src.tasks.analysis import analyze_track
        try:
            result = analyze_track.delay(str(self.id))
            return result.id
        except Exception as e:
            self.status = self.Status.FAILED
            self.status_message = f"Failed to queue analysis: {e}"
            self.save(update_fields=["status", "status_message"])
            return None


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
