# tests/test_api/test_models.py
import pytest
from unittest.mock import patch, MagicMock
from django.test import TestCase
from src.api.models import Track


class TestTrackModel(TestCase):
    """Tests for Track model methods."""

    def test_get_or_none_returns_track(self):
        """get_or_none returns track when it exists."""
        track = Track.objects.create(
            original_filename="test.wav",
            storage_path="test.wav",
            file_size=1000,
        )
        result = Track.get_or_none(str(track.id))
        assert result == track

    def test_get_or_none_returns_none_for_missing(self):
        """get_or_none returns None when track doesn't exist."""
        result = Track.get_or_none("00000000-0000-0000-0000-000000000000")
        assert result is None

    def test_is_ready_true_when_ready(self):
        """is_ready returns True when status is READY."""
        track = Track.objects.create(
            original_filename="test.wav",
            storage_path="test.wav",
            file_size=1000,
            status=Track.Status.READY,
        )
        assert track.is_ready is True

    def test_is_ready_false_when_pending(self):
        """is_ready returns False when status is not READY."""
        track = Track.objects.create(
            original_filename="test.wav",
            storage_path="test.wav",
            file_size=1000,
            status=Track.Status.PENDING,
        )
        assert track.is_ready is False

    def test_has_analysis_true_when_present(self):
        """has_analysis returns True when analysis exists."""
        track = Track.objects.create(
            original_filename="test.wav",
            storage_path="test.wav",
            file_size=1000,
            analysis={"rhythm": {"bpm": 120}},
        )
        assert track.has_analysis("rhythm") is True

    def test_has_analysis_false_when_missing(self):
        """has_analysis returns False when analysis doesn't exist."""
        track = Track.objects.create(
            original_filename="test.wav",
            storage_path="test.wav",
            file_size=1000,
            analysis={},
        )
        assert track.has_analysis("rhythm") is False

    def test_has_analysis_false_when_error(self):
        """has_analysis returns False when analysis contains error."""
        track = Track.objects.create(
            original_filename="test.wav",
            storage_path="test.wav",
            file_size=1000,
            analysis={"rhythm": {"error": "Failed"}},
        )
        assert track.has_analysis("rhythm") is False

    def test_get_analysis_returns_data(self):
        """get_analysis returns analysis data."""
        track = Track.objects.create(
            original_filename="test.wav",
            storage_path="test.wav",
            file_size=1000,
            analysis={"rhythm": {"bpm": 120}},
        )
        result = track.get_analysis("rhythm")
        assert result == {"bpm": 120}

    def test_get_analysis_raises_key_error_when_missing(self):
        """get_analysis raises KeyError when analysis doesn't exist."""
        track = Track.objects.create(
            original_filename="test.wav",
            storage_path="test.wav",
            file_size=1000,
            analysis={},
        )
        with pytest.raises(KeyError):
            track.get_analysis("rhythm")

    def test_get_analysis_raises_value_error_on_error(self):
        """get_analysis raises ValueError when analysis has error."""
        track = Track.objects.create(
            original_filename="test.wav",
            storage_path="test.wav",
            file_size=1000,
            analysis={"rhythm": {"error": "Analysis failed"}},
        )
        with pytest.raises(ValueError, match="Analysis failed"):
            track.get_analysis("rhythm")

    def test_set_analysis_stores_data(self):
        """set_analysis stores analysis and saves to DB."""
        track = Track.objects.create(
            original_filename="test.wav",
            storage_path="test.wav",
            file_size=1000,
        )
        track.set_analysis("rhythm", {"bpm": 120})
        track.refresh_from_db()
        assert track.analysis["rhythm"] == {"bpm": 120}

    def test_queue_analysis_returns_task_id(self):
        """queue_analysis returns task ID on success."""
        track = Track.objects.create(
            original_filename="test.wav",
            storage_path="test.wav",
            file_size=1000,
        )
        with patch("src.tasks.analysis.analyze_track") as mock_task:
            mock_result = MagicMock()
            mock_result.id = "task-123"
            mock_task.delay.return_value = mock_result

            result = track.queue_analysis()

            assert result == "task-123"
            mock_task.delay.assert_called_once_with(str(track.id))

    def test_queue_analysis_sets_failed_status_on_error(self):
        """queue_analysis sets FAILED status when queueing fails."""
        track = Track.objects.create(
            original_filename="test.wav",
            storage_path="test.wav",
            file_size=1000,
            status=Track.Status.PENDING,
        )
        with patch("src.tasks.analysis.analyze_track") as mock_task:
            mock_task.delay.side_effect = Exception("Broker unavailable")

            result = track.queue_analysis()

            assert result is None
            track.refresh_from_db()
            assert track.status == Track.Status.FAILED
            assert "Broker unavailable" in track.status_message

    def test_file_path_returns_filesystem_path_when_available(self):
        """file_path returns filesystem path from storage backend."""
        track = Track.objects.create(
            original_filename="test.wav",
            storage_path="uploads/test.wav",
            file_url="https://cdn.example.com/test.wav",
            file_size=1000,
        )
        with patch("django.core.files.storage.default_storage") as mock_storage:
            mock_storage.path.return_value = "/var/uploads/test.wav"

            result = track.file_path

            assert result == "/var/uploads/test.wav"

    def test_file_path_falls_back_to_url_when_path_not_implemented(self):
        """file_path returns file_url when storage has no path method."""
        track = Track.objects.create(
            original_filename="test.wav",
            storage_path="uploads/test.wav",
            file_url="https://cdn.example.com/test.wav",
            file_size=1000,
        )
        with patch("django.core.files.storage.default_storage") as mock_storage:
            mock_storage.path.side_effect = NotImplementedError()

            result = track.file_path

            assert result == "https://cdn.example.com/test.wav"
