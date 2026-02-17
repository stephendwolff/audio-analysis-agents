"""End-to-end integration test for the sketchbook API flow."""

import uuid
from unittest.mock import patch, MagicMock
from django.test import TestCase
from django.contrib.auth.models import User
from rest_framework.test import APIClient
from src.sketchbook.models import Fragment


class TestAnalyseEndToEnd(TestCase):
    """Test the full POST -> task -> GET polling flow."""

    def setUp(self):
        self.user = User.objects.create_user(username="test", password="test")
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)

    @patch("src.sketchbook.views.analyse_fragment")
    def test_post_then_task_then_get(self, mock_task):
        """POST creates fragment, simulated task completes it, GET returns result."""
        mock_task.delay.return_value = MagicMock(id="task-1")
        fid = uuid.uuid4()

        from django.core.files.uploadedfile import SimpleUploadedFile
        audio = SimpleUploadedFile("test.wav", b"data", content_type="audio/wav")

        # POST — should return 202
        post_resp = self.client.post(
            "/api/analyse/",
            {"audio": audio, "mime_type": "audio/wav",
             "fragment_id": str(fid), "duration_seconds": "5.0"},
            format="multipart",
        )
        assert post_resp.status_code == 202
        post_data = post_resp.json()
        assert "poll_url" in post_data
        assert post_data["poll_url"] == f"/api/analyse/{fid}/"

        # GET while pending — should return pending
        get_resp = self.client.get(f"/api/analyse/{fid}/")
        assert get_resp.status_code == 200
        assert get_resp.json()["status"] == "pending"

        # Simulate task completing
        fragment = Fragment.objects.get(fragment_id=fid)
        fragment.status = Fragment.Status.COMPLETE
        fragment.analysis = {
            "dimensions": {"bpm": 120, "time_signature": "4/4",
                          "swing": 0.1, "steadiness": 0.9, "upbeat": False},
            "descriptors": ["moderate-tempo", "straight", "steady"],
            "raw_data": {"beats": [0.5], "onsets": [0.1],
                        "spectral_centroid_mean": 2000, "rms_energy_mean": 0.05},
        }
        fragment.save()

        # GET after completion — should return full result
        get_resp = self.client.get(f"/api/analyse/{fid}/")
        assert get_resp.status_code == 200
        data = get_resp.json()
        assert data["status"] == "complete"
        assert data["dimensions"]["bpm"] == 120
        assert "descriptors" in data
        assert "raw_data" in data

        # Re-POST same fragment_id — should return 200 with cached result
        audio2 = SimpleUploadedFile("test.wav", b"data", content_type="audio/wav")
        repost_resp = self.client.post(
            "/api/analyse/",
            {"audio": audio2, "mime_type": "audio/wav",
             "fragment_id": str(fid), "duration_seconds": "5.0"},
            format="multipart",
        )
        assert repost_resp.status_code == 200
        assert repost_resp.json()["dimensions"]["bpm"] == 120
