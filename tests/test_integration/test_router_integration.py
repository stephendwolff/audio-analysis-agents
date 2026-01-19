# tests/test_integration/test_router_integration.py
"""Integration tests for the full router flow."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from asgiref.sync import sync_to_async
from django.test import TestCase
from src.api.models import Track
from src.orchestrator.providers import CachedAnalysisProvider
from src.orchestrator.llm_routing_litellm import LLMRouterLiteLLM


class TestRouterIntegration(TestCase):
    """Integration tests for the full router flow."""

    @pytest.mark.asyncio
    async def test_full_flow_with_cached_data(self):
        """Test router with real cached data from database."""
        # Create track with analysis (wrap in sync_to_async for async test)
        @sync_to_async
        def create_track():
            return Track.objects.create(
                original_filename="test.wav",
                storage_path="test.wav",
                file_size=1000,
                status=Track.Status.READY,
                analysis={
                    "rhythm": {"bpm": 120, "beat_count": 48},
                    "spectral": {"centroid": 2000},
                },
            )

        track = await create_track()

        # Mock LiteLLM to request tool call
        mock_tool_response = MagicMock()
        mock_tool_response.choices = [MagicMock()]
        mock_function = MagicMock()
        mock_function.name = "analyse_rhythm"
        mock_function.arguments = f'{{"track_id": "{track.id}"}}'
        mock_tool_response.choices[0].message.tool_calls = [
            MagicMock(id="call_123", function=mock_function)
        ]
        mock_tool_response.choices[0].message.content = None
        mock_tool_response.choices[0].message.model_dump = MagicMock(return_value={
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_123",
                "type": "function",
                "function": {"name": "analyse_rhythm", "arguments": f'{{"track_id": "{track.id}"}}'}
            }]
        })

        # Mock final response (no tool calls)
        mock_final_response = MagicMock()
        mock_final_response.choices = [MagicMock()]
        mock_final_response.choices[0].message.tool_calls = None
        mock_final_response.choices[0].message.content = "The tempo is 120 BPM"

        with patch("src.orchestrator.llm_routing_litellm.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=[
                mock_tool_response,
                mock_final_response,
            ])

            provider = CachedAnalysisProvider()
            router = LLMRouterLiteLLM(
                provider=provider,
                model="test-model",
                enable_tracing=False
            )

            events = []
            async for event in router.process_question_stream(
                "What's the tempo?",
                str(track.id)
            ):
                events.append(event)

            # Verify tool_call event
            tool_calls = [e for e in events if e["type"] == "tool_call"]
            assert len(tool_calls) == 1
            assert tool_calls[0]["tool"] == "analyse_rhythm"

            # Verify tool_result got real data from DB (success=True means data was found)
            tool_results = [e for e in events if e["type"] == "tool_result"]
            assert len(tool_results) == 1
            assert tool_results[0]["success"] is True

            # Verify done event
            done_events = [e for e in events if e["type"] == "done"]
            assert len(done_events) == 1
            assert done_events[0]["full_response"] == "The tempo is 120 BPM"

    @pytest.mark.asyncio
    async def test_provider_fetches_from_database(self):
        """Test that CachedAnalysisProvider correctly fetches data from Track model."""
        @sync_to_async
        def create_track():
            return Track.objects.create(
                original_filename="test.wav",
                storage_path="test.wav",
                file_size=1000,
                status=Track.Status.READY,
                analysis={
                    "rhythm": {"bpm": 145, "confidence": 0.95},
                },
            )

        track = await create_track()
        provider = CachedAnalysisProvider()

        # Test has_analysis (wrap sync calls)
        has_rhythm = await sync_to_async(provider.has_analysis)("rhythm", str(track.id))
        has_spectral = await sync_to_async(provider.has_analysis)("spectral", str(track.id))
        assert has_rhythm is True
        assert has_spectral is False

        # Test get_analysis
        result = await sync_to_async(provider.get_analysis)("rhythm", str(track.id))
        assert result["bpm"] == 145
        assert result["confidence"] == 0.95
