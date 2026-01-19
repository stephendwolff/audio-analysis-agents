# tests/test_orchestrator/test_llm_routing_litellm.py
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from src.orchestrator.llm_routing_litellm import LLMRouterLiteLLM
from src.orchestrator.providers import AnalysisProvider


class TestLLMRouterLiteLLM:
    """Tests for LLMRouterLiteLLM."""

    @pytest.mark.asyncio
    async def test_yields_tool_call_event(self):
        """Router yields tool_call event when LLM requests tool."""
        provider = Mock(spec=AnalysisProvider)
        provider.get_analysis = Mock(return_value={"bpm": 120})

        # Mock LiteLLM response with tool call
        mock_tool_response = MagicMock()
        mock_tool_response.choices = [MagicMock()]
        mock_function = MagicMock()
        mock_function.name = "analyse_rhythm"
        mock_function.arguments = '{"track_id": "track-1"}'
        mock_tool_response.choices[0].message.tool_calls = [
            MagicMock(id="call_123", function=mock_function)
        ]
        mock_tool_response.choices[0].message.content = None
        mock_tool_response.choices[0].message.model_dump = Mock(return_value={
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_123",
                "type": "function",
                "function": {"name": "analyse_rhythm", "arguments": '{"track_id": "track-1"}'}
            }]
        })

        # Mock response without tool calls (triggers streaming path)
        mock_no_tool_response = MagicMock()
        mock_no_tool_response.choices = [MagicMock()]
        mock_no_tool_response.choices[0].message.tool_calls = None
        mock_no_tool_response.choices[0].message.content = None

        # Mock streaming response (async generator)
        async def mock_stream():
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = "Done"
            yield chunk

        with patch("src.orchestrator.llm_routing_litellm.litellm") as mock_litellm:
            # First call returns tool_response, second call (after tool execution) returns no_tool response
            # Third call (streaming) returns the generator
            mock_litellm.acompletion = AsyncMock(side_effect=[
                mock_tool_response,
                mock_no_tool_response,
                mock_stream(),
            ])

            router = LLMRouterLiteLLM(provider=provider, model="test-model", enable_tracing=False)

            events = []
            async for event in router.process_question_stream("What's the tempo?", "track-1"):
                events.append(event)

            # Should have tool_call, tool_result, tokens, done
            event_types = [e["type"] for e in events]
            assert "tool_call" in event_types
            assert "tool_result" in event_types

    @pytest.mark.asyncio
    async def test_uses_provider_for_analysis(self):
        """Router uses provider.get_analysis for tool results."""
        provider = Mock(spec=AnalysisProvider)
        provider.get_analysis = Mock(return_value={"bpm": 120})

        mock_tool_response = MagicMock()
        mock_tool_response.choices = [MagicMock()]
        mock_function = MagicMock()
        mock_function.name = "analyse_rhythm"
        mock_function.arguments = '{"track_id": "track-1"}'
        mock_tool_response.choices[0].message.tool_calls = [
            MagicMock(id="call_123", function=mock_function)
        ]
        mock_tool_response.choices[0].message.content = None
        mock_tool_response.choices[0].message.model_dump = Mock(return_value={
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_123",
                "type": "function",
                "function": {"name": "analyse_rhythm", "arguments": '{"track_id": "track-1"}'}
            }]
        })

        # Mock response without tool calls
        mock_no_tool_response = MagicMock()
        mock_no_tool_response.choices = [MagicMock()]
        mock_no_tool_response.choices[0].message.tool_calls = None
        mock_no_tool_response.choices[0].message.content = None

        async def mock_stream():
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = "Done"
            yield chunk

        with patch("src.orchestrator.llm_routing_litellm.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=[
                mock_tool_response,
                mock_no_tool_response,
                mock_stream(),
            ])

            router = LLMRouterLiteLLM(provider=provider, model="test-model", enable_tracing=False)

            events = []
            async for event in router.process_question_stream("What's the tempo?", "track-1"):
                events.append(event)

            # Provider should have been called
            provider.get_analysis.assert_called_with("rhythm", "track-1")
