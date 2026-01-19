# tests/test_orchestrator/test_providers.py
import pytest
from django.test import TestCase
from src.orchestrator.providers import (
    AnalysisProvider,
    CachedAnalysisProvider,
)
from src.api.models import Track


class TestCachedAnalysisProvider(TestCase):
    """Tests for CachedAnalysisProvider."""

    def test_get_analysis_returns_cached_data(self):
        """get_analysis returns data from track.analysis."""
        track = Track.objects.create(
            original_filename="test.wav",
            storage_path="test.wav",
            file_size=1000,
            analysis={"rhythm": {"bpm": 120}},
        )
        provider = CachedAnalysisProvider()
        result = provider.get_analysis("rhythm", str(track.id))
        assert result == {"bpm": 120}

    def test_get_analysis_raises_on_missing(self):
        """get_analysis raises KeyError when analysis missing."""
        track = Track.objects.create(
            original_filename="test.wav",
            storage_path="test.wav",
            file_size=1000,
            analysis={},
        )
        provider = CachedAnalysisProvider()
        with pytest.raises(KeyError):
            provider.get_analysis("rhythm", str(track.id))

    def test_get_analysis_raises_on_error(self):
        """get_analysis raises ValueError when analysis has error."""
        track = Track.objects.create(
            original_filename="test.wav",
            storage_path="test.wav",
            file_size=1000,
            analysis={"rhythm": {"error": "Failed"}},
        )
        provider = CachedAnalysisProvider()
        with pytest.raises(ValueError, match="Failed"):
            provider.get_analysis("rhythm", str(track.id))

    def test_has_analysis_true_when_present(self):
        """has_analysis returns True when analysis exists."""
        track = Track.objects.create(
            original_filename="test.wav",
            storage_path="test.wav",
            file_size=1000,
            analysis={"rhythm": {"bpm": 120}},
        )
        provider = CachedAnalysisProvider()
        assert provider.has_analysis("rhythm", str(track.id)) is True

    def test_has_analysis_false_when_missing(self):
        """has_analysis returns False when analysis missing."""
        track = Track.objects.create(
            original_filename="test.wav",
            storage_path="test.wav",
            file_size=1000,
            analysis={},
        )
        provider = CachedAnalysisProvider()
        assert provider.has_analysis("rhythm", str(track.id)) is False

    def test_has_analysis_false_for_nonexistent_track(self):
        """has_analysis returns False for nonexistent track."""
        provider = CachedAnalysisProvider()
        assert provider.has_analysis("rhythm", "00000000-0000-0000-0000-000000000000") is False
