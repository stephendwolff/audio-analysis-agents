"""Tests for the sketchbook API views."""

import uuid
import pytest
from unittest.mock import patch, MagicMock
from django.test import TestCase
from django.contrib.auth.models import User
from rest_framework.test import APIClient
from src.sketchbook.models import Fragment


class TestHealthView(TestCase):

    def test_health_returns_ok(self):
        client = APIClient()
        response = client.get("/api/health/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "agents" in data
        assert "rhythm" in data["agents"]


class TestAnalysePostView(TestCase):

    def setUp(self):
        self.user = User.objects.create_user(username="test", password="test")
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)

    @patch("src.sketchbook.views.analyse_fragment")
    def test_post_returns_202(self, mock_task):
        mock_task.delay.return_value = MagicMock(id="task-123")
        from django.core.files.uploadedfile import SimpleUploadedFile

        audio = SimpleUploadedFile("test.wav", b"fake-audio-data", content_type="audio/wav")
        response = self.client.post(
            "/api/analyse/",
            {"audio": audio, "mime_type": "audio/wav",
             "fragment_id": str(uuid.uuid4()), "duration_seconds": "5.0"},
            format="multipart",
        )
        assert response.status_code == 202
        data = response.json()
        assert data["status"] == "pending"
        assert "poll_url" in data

    def test_post_missing_audio_returns_400(self):
        response = self.client.post(
            "/api/analyse/",
            {"mime_type": "audio/wav",
             "fragment_id": str(uuid.uuid4()), "duration_seconds": "5.0"},
            format="multipart",
        )
        assert response.status_code == 400

    def test_post_unsupported_mime_returns_422(self):
        from django.core.files.uploadedfile import SimpleUploadedFile

        audio = SimpleUploadedFile("test.xyz", b"data", content_type="audio/xyz")
        response = self.client.post(
            "/api/analyse/",
            {"audio": audio, "mime_type": "audio/xyz",
             "fragment_id": str(uuid.uuid4()), "duration_seconds": "5.0"},
            format="multipart",
        )
        assert response.status_code == 422

    @patch("src.sketchbook.views.analyse_fragment")
    def test_post_existing_complete_fragment_returns_200(self, mock_task):
        fid = uuid.uuid4()
        Fragment.objects.create(
            fragment_id=fid, user=self.user,
            audio_storage_path="test.wav", mime_type="audio/wav",
            duration_seconds=5.0, model_version="1.0.0",
            status=Fragment.Status.COMPLETE,
            analysis={"dimensions": {"bpm": 120}, "descriptors": [], "raw_data": {}},
        )
        from django.core.files.uploadedfile import SimpleUploadedFile

        audio = SimpleUploadedFile("test.wav", b"data", content_type="audio/wav")
        response = self.client.post(
            "/api/analyse/",
            {"audio": audio, "mime_type": "audio/wav",
             "fragment_id": str(fid), "duration_seconds": "5.0"},
            format="multipart",
        )
        assert response.status_code == 200
        assert response.json()["dimensions"]["bpm"] == 120

    @patch("src.sketchbook.views.analyse_fragment")
    def test_post_existing_pending_fragment_returns_202(self, mock_task):
        fid = uuid.uuid4()
        Fragment.objects.create(
            fragment_id=fid, user=self.user,
            audio_storage_path="test.wav", mime_type="audio/wav",
            duration_seconds=5.0, model_version="1.0.0",
            status=Fragment.Status.PENDING,
        )
        from django.core.files.uploadedfile import SimpleUploadedFile

        audio = SimpleUploadedFile("test.wav", b"data", content_type="audio/wav")
        response = self.client.post(
            "/api/analyse/",
            {"audio": audio, "mime_type": "audio/wav",
             "fragment_id": str(fid), "duration_seconds": "5.0"},
            format="multipart",
        )
        assert response.status_code == 202

    def test_post_requires_auth(self):
        client = APIClient()  # unauthenticated
        from django.core.files.uploadedfile import SimpleUploadedFile

        audio = SimpleUploadedFile("test.wav", b"data", content_type="audio/wav")
        response = client.post(
            "/api/analyse/",
            {"audio": audio, "mime_type": "audio/wav",
             "fragment_id": str(uuid.uuid4()), "duration_seconds": "5.0"},
            format="multipart",
        )
        assert response.status_code in (401, 403)


class TestAnalyseGetView(TestCase):

    def setUp(self):
        self.user = User.objects.create_user(username="test", password="test")
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)

    def test_get_complete_fragment(self):
        fid = uuid.uuid4()
        Fragment.objects.create(
            fragment_id=fid, user=self.user,
            audio_storage_path="test.wav", mime_type="audio/wav",
            duration_seconds=5.0, model_version="1.0.0",
            status=Fragment.Status.COMPLETE,
            analysis={"dimensions": {"bpm": 120}, "descriptors": ["steady"], "raw_data": {}},
        )
        response = self.client.get(f"/api/analyse/{fid}/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "complete"
        assert data["dimensions"]["bpm"] == 120
        assert data["model_version"] == "1.0.0"

    def test_get_pending_fragment(self):
        fid = uuid.uuid4()
        Fragment.objects.create(
            fragment_id=fid, user=self.user,
            audio_storage_path="test.wav", mime_type="audio/wav",
            duration_seconds=5.0, model_version="1.0.0",
            status=Fragment.Status.PENDING,
        )
        response = self.client.get(f"/api/analyse/{fid}/")
        assert response.status_code == 200
        assert response.json()["status"] == "pending"

    def test_get_failed_fragment(self):
        fid = uuid.uuid4()
        Fragment.objects.create(
            fragment_id=fid, user=self.user,
            audio_storage_path="test.wav", mime_type="audio/wav",
            duration_seconds=5.0, model_version="1.0.0",
            status=Fragment.Status.FAILED, error_message="Timeout",
        )
        response = self.client.get(f"/api/analyse/{fid}/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        assert data["error"] == "Timeout"

    def test_get_unknown_fragment_returns_404(self):
        response = self.client.get(f"/api/analyse/{uuid.uuid4()}/")
        assert response.status_code == 404
