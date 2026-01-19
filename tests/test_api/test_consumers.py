# tests/test_api/test_consumers.py
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from django.test import TestCase
from asgiref.sync import sync_to_async
from src.api.consumers import ChatConsumer
from src.api.models import Track


class TestChatConsumer(TestCase):
    """Tests for ChatConsumer."""

    @pytest.mark.asyncio
    @pytest.mark.django_db(transaction=True)
    async def test_process_question_streams_events(self):
        """process_question streams router events to client."""
        # Create a ready track using sync_to_async
        @sync_to_async
        def create_track():
            return Track.objects.create(
                original_filename="test.wav",
                storage_path="test.wav",
                file_size=1000,
                status=Track.Status.READY,
                analysis={"rhythm": {"bpm": 120}},
            )

        track = await create_track()

        # Mock router to yield events
        async def mock_stream(question, track_id):
            yield {"type": "tool_call", "tool": "analyse_rhythm"}
            yield {"type": "tool_result", "tool": "analyse_rhythm", "success": True}
            yield {"type": "token", "text": "The tempo"}
            yield {"type": "done", "full_response": "The tempo is 120 BPM"}

        mock_router = MagicMock()
        mock_router.process_question_stream = mock_stream

        with patch("src.orchestrator.llm_routing_litellm.LLMRouterLiteLLM", return_value=mock_router):
            with patch("src.orchestrator.providers.CachedAnalysisProvider"):
                consumer = ChatConsumer()
                consumer.track = track
                consumer.track_id = str(track.id)
                consumer.send_json = AsyncMock()

                await consumer.process_question("What's the tempo?")

                # Should have sent multiple events
                assert consumer.send_json.call_count >= 4

                # Check event types
                calls = consumer.send_json.call_args_list
                event_types = [call[0][0]["type"] for call in calls]
                assert "tool_call" in event_types
                assert "done" in event_types
