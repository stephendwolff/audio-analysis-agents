"""Tests for the analyse_fragment Celery task."""

import uuid
import pytest
from unittest.mock import patch, MagicMock
from django.test import TestCase
from django.contrib.auth.models import User
from src.sketchbook.models import Fragment
from src.sketchbook.tasks import analyse_fragment
from src.agents.base import AnalysisResult


class TestAnalyseFragmentTask(TestCase):

    def setUp(self):
        self.user = User.objects.create_user(username="test", password="test")
        self.fragment = Fragment.objects.create(
            fragment_id=uuid.uuid4(),
            user=self.user,
            audio_storage_path="/tmp/test.wav",
            mime_type="audio/wav",
            duration_seconds=5.0,
            model_version="1.0.0",
        )

    @patch("src.sketchbook.tasks.TemporalAgent")
    @patch("src.sketchbook.tasks.SpectralAgent")
    @patch("src.sketchbook.tasks.RhythmAgent")
    @patch("src.sketchbook.tasks.load_audio")
    def test_sets_status_to_complete_on_success(
        self, mock_load, mock_rhythm_cls, mock_spectral_cls, mock_temporal_cls
    ):
        mock_audio = MagicMock()
        mock_audio.samples = MagicMock()
        mock_audio.sample_rate = 22050
        mock_load.return_value = mock_audio

        mock_rhythm_cls.return_value.analyse.return_value = AnalysisResult(
            agent="rhythm", success=True, data={
                "tempo_bpm": 120, "time_signature": "4/4", "swing": 0.1,
                "steadiness": 0.9, "upbeat": False,
                "beat_times": [0.5, 1.0], "onset_times": [0.1, 0.5],
            }
        )
        mock_spectral_cls.return_value.analyse.return_value = AnalysisResult(
            agent="spectral", success=True,
            data={"spectral_centroid": {"mean": 2000}},
        )
        mock_temporal_cls.return_value.analyse.return_value = AnalysisResult(
            agent="temporal", success=True,
            data={"rms_energy": {"mean": 0.05}},
        )

        analyse_fragment(str(self.fragment.fragment_id))

        self.fragment.refresh_from_db()
        assert self.fragment.status == Fragment.Status.COMPLETE
        assert "dimensions" in self.fragment.analysis
        assert self.fragment.analysis["dimensions"]["bpm"] == 120

    @patch("src.sketchbook.tasks.load_audio")
    def test_sets_status_to_failed_on_error(self, mock_load):
        mock_load.side_effect = FileNotFoundError("audio not found")

        analyse_fragment(str(self.fragment.fragment_id))

        self.fragment.refresh_from_db()
        assert self.fragment.status == Fragment.Status.FAILED
        assert "audio not found" in self.fragment.error_message

    @patch("src.sketchbook.tasks.TemporalAgent")
    @patch("src.sketchbook.tasks.SpectralAgent")
    @patch("src.sketchbook.tasks.RhythmAgent")
    @patch("src.sketchbook.tasks.load_audio")
    def test_sets_status_to_analyzing_during_run(
        self, mock_load, mock_rhythm_cls, mock_spectral_cls, mock_temporal_cls
    ):
        statuses = []

        def capture_status(*args, **kwargs):
            self.fragment.refresh_from_db()
            statuses.append(self.fragment.status)
            return AnalysisResult(
                agent="rhythm", success=True, data={
                    "tempo_bpm": 120, "time_signature": "4/4", "swing": 0.0,
                    "steadiness": 0.9, "upbeat": False,
                    "beat_times": [], "onset_times": [],
                }
            )

        mock_audio = MagicMock()
        mock_load.return_value = mock_audio
        mock_rhythm_cls.return_value.analyse.side_effect = capture_status
        mock_spectral_cls.return_value.analyse.return_value = AnalysisResult(
            agent="spectral", success=True, data={"spectral_centroid": {"mean": 0}},
        )
        mock_temporal_cls.return_value.analyse.return_value = AnalysisResult(
            agent="temporal", success=True, data={"rms_energy": {"mean": 0}},
        )

        analyse_fragment(str(self.fragment.fragment_id))
        assert "analyzing" in statuses
