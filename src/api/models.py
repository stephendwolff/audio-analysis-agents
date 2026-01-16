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
