# tests/test_api/test_views.py
import pytest
from django.test import TestCase
from src.api import views


class TestUtilityFunctionsRemoved(TestCase):
    """Tests that utility functions were moved to model."""

    def test_get_track_not_in_views(self):
        """get_track function should not exist in views."""
        assert not hasattr(views, 'get_track'), "get_track should be moved to Track.get_or_none"

    def test_get_track_path_not_in_views(self):
        """get_track_path function should not exist in views."""
        assert not hasattr(views, 'get_track_path'), "get_track_path should be moved to Track.file_path"
