import uuid
import pytest
from django.test import TestCase
from django.contrib.auth.models import User
from src.sketchbook.models import Fragment


class TestFragmentModel(TestCase):

    def setUp(self):
        self.user = User.objects.create_user(username="test", password="test")

    def test_create_fragment(self):
        fid = uuid.uuid4()
        fragment = Fragment.objects.create(
            fragment_id=fid,
            user=self.user,
            audio_storage_path="test.wav",
            mime_type="audio/wav",
            duration_seconds=5.0,
            model_version="1.0.0",
        )
        assert fragment.fragment_id == fid
        assert fragment.status == Fragment.Status.PENDING

    def test_fragment_id_is_unique(self):
        from django.db import IntegrityError
        fid = uuid.uuid4()
        Fragment.objects.create(
            fragment_id=fid,
            user=self.user,
            audio_storage_path="a.wav",
            mime_type="audio/wav",
            duration_seconds=5.0,
            model_version="1.0.0",
        )
        with pytest.raises(IntegrityError):
            Fragment.objects.create(
                fragment_id=fid,
                user=self.user,
                audio_storage_path="b.wav",
                mime_type="audio/wav",
                duration_seconds=5.0,
                model_version="1.0.0",
            )

    def test_default_status_is_pending(self):
        fragment = Fragment.objects.create(
            fragment_id=uuid.uuid4(),
            user=self.user,
            audio_storage_path="test.wav",
            mime_type="audio/wav",
            duration_seconds=5.0,
            model_version="1.0.0",
        )
        assert fragment.status == "pending"

    def test_status_choices(self):
        assert Fragment.Status.PENDING == "pending"
        assert Fragment.Status.ANALYZING == "analyzing"
        assert Fragment.Status.COMPLETE == "complete"
        assert Fragment.Status.FAILED == "failed"
