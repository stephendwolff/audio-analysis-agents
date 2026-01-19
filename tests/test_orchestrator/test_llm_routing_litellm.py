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

        # Mock response without tool calls (returns content directly, no streaming)
        mock_no_tool_response = MagicMock()
        mock_no_tool_response.choices = [MagicMock()]
        mock_no_tool_response.choices[0].message.tool_calls = None
        mock_no_tool_response.choices[0].message.content = "Done"

        with patch("src.orchestrator.llm_routing_litellm.litellm") as mock_litellm:
            # First call returns tool_response, second call (after tool execution) returns content
            mock_litellm.acompletion = AsyncMock(side_effect=[
                mock_tool_response,
                mock_no_tool_response,
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

        # Mock response without tool calls (returns content directly, no streaming)
        mock_no_tool_response = MagicMock()
        mock_no_tool_response.choices = [MagicMock()]
        mock_no_tool_response.choices[0].message.tool_calls = None
        mock_no_tool_response.choices[0].message.content = "Done"

        with patch("src.orchestrator.llm_routing_litellm.litellm") as mock_litellm:
            # First call returns tool_response, second call (after tool execution) returns content
            mock_litellm.acompletion = AsyncMock(side_effect=[
                mock_tool_response,
                mock_no_tool_response,
            ])

            router = LLMRouterLiteLLM(provider=provider, model="test-model", enable_tracing=False)

            events = []
            async for event in router.process_question_stream("What's the tempo?", "track-1"):
                events.append(event)

            # Provider should have been called
            provider.get_analysis.assert_called_with("rhythm", "track-1")

    @pytest.mark.asyncio
    async def test_yields_error_event_on_llm_exception(self):
        """Router yields error event when LLM call raises exception."""
        provider = Mock(spec=AnalysisProvider)

        with patch("src.orchestrator.llm_routing_litellm.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=ValueError("LLM API failed"))

            router = LLMRouterLiteLLM(provider=provider, model="test-model", enable_tracing=False)

            events = []
            async for event in router.process_question_stream("question", "track-1"):
                events.append(event)

            # Should have error event
            error_events = [e for e in events if e["type"] == "error"]
            assert len(error_events) == 1
            assert "LLM API failed" in error_events[0]["message"]

    @pytest.mark.asyncio
    async def test_yields_done_event_with_full_response(self):
        """Router yields done event with complete response."""
        provider = Mock(spec=AnalysisProvider)

        # Mock response with no tool calls (just content)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].message.content = "The answer is 42"

        with patch("src.orchestrator.llm_routing_litellm.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)

            router = LLMRouterLiteLLM(provider=provider, model="test-model", enable_tracing=False)

            events = []
            async for event in router.process_question_stream("question", "track-1"):
                events.append(event)

            # Should have done event with full response
            done_events = [e for e in events if e["type"] == "done"]
            assert len(done_events) == 1
            assert done_events[0]["full_response"] == "The answer is 42"

            # Should also have token event with the content
            token_events = [e for e in events if e["type"] == "token"]
            assert len(token_events) == 1
            assert token_events[0]["text"] == "The answer is 42"
