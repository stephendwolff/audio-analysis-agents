"""
Base router class for LLM routing.

All router implementations inherit from BaseRouter and implement
the process_question_stream method for async event streaming.
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator

from .providers import AnalysisProvider
from .events import RouterEvent


class BaseRouter(ABC):
    """
    Abstract base class for LLM routers.

    Routers connect user questions to analysis tools via an LLM.
    They yield events during processing for real-time streaming.

    All routers accept an AnalysisProvider for fetching analysis data,
    allowing the same router to work with cached or live data.
    """

    def __init__(
        self,
        provider: AnalysisProvider,
        enable_tracing: bool = True,
        project_name: str = "audio-analysis-agents",
    ):
        """
        Initialize the router.

        Args:
            provider: Strategy for fetching analysis data
            enable_tracing: Whether to enable Opik tracing
            project_name: Project name for Opik
        """
        self.provider = provider
        self.enable_tracing = enable_tracing
        self.project_name = project_name

        if enable_tracing:
            from src.observability.tracing import init_tracing
            init_tracing(project_name)

    @abstractmethod
    async def process_question_stream(
        self,
        question: str,
        track_id: str,
    ) -> AsyncIterator[RouterEvent]:
        """
        Process a question, yielding events as they occur.

        Args:
            question: The user's question
            track_id: UUID of the track to analyse

        Yields:
            RouterEvent dictionaries (tool_call, tool_result, token, done, error)
        """
        ...

    async def process_question(self, question: str, track_id: str) -> dict:
        """
        Process a question in batch mode.

        Collects all events and returns the final result.
        Useful for non-streaming contexts like tests or CLI.

        Args:
            question: The user's question
            track_id: UUID of the track to analyse

        Returns:
            Dict with 'response' and 'tools_called'
        """
        tools_called = []
        full_response = ""

        async for event in self.process_question_stream(question, track_id):
            if event["type"] == "tool_call":
                tools_called.append(event["tool"])
            elif event["type"] == "token":
                full_response += event["text"]
            elif event["type"] == "done":
                full_response = event["full_response"]
            elif event["type"] == "error":
                return {"error": event["message"], "tools_called": tools_called}

        return {
            "response": full_response,
            "tools_called": tools_called,
        }
