# tests/test_orchestrator/test_base_router.py
import pytest
from unittest.mock import Mock, AsyncMock
from src.orchestrator.base import BaseRouter
from src.orchestrator.providers import AnalysisProvider
from src.orchestrator.events import RouterEvent


class TestBaseRouter:
    """Tests for BaseRouter ABC."""

    def test_cannot_instantiate_directly(self):
        """BaseRouter cannot be instantiated directly."""
        provider = Mock(spec=AnalysisProvider)
        with pytest.raises(TypeError):
            BaseRouter(provider=provider)

    def test_subclass_must_implement_process_question_stream(self):
        """Subclass must implement process_question_stream."""
        provider = Mock(spec=AnalysisProvider)

        class IncompleteRouter(BaseRouter):
            pass

        with pytest.raises(TypeError):
            IncompleteRouter(provider=provider)

    @pytest.mark.asyncio
    async def test_process_question_collects_events(self):
        """process_question collects events and returns result."""
        provider = Mock(spec=AnalysisProvider)

        class TestRouter(BaseRouter):
            async def process_question_stream(self, question, track_id):
                yield {"type": "tool_call", "tool": "analyse_rhythm"}
                yield {"type": "tool_result", "tool": "analyse_rhythm", "success": True}
                yield {"type": "token", "text": "The "}
                yield {"type": "token", "text": "tempo"}
                yield {"type": "done", "full_response": "The tempo is 120 BPM"}

        router = TestRouter(provider=provider, enable_tracing=False)
        result = await router.process_question("What's the tempo?", "track-123")

        assert result["response"] == "The tempo is 120 BPM"
        assert "analyse_rhythm" in result["tools_called"]
