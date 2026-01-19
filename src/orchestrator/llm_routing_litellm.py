"""
LLM Routing Module - LiteLLM Provider-Agnostic Implementation

Uses LiteLLM for provider abstraction - same code works with Claude, Gemini, GPT-4.
Async implementation with streaming for real-time WebSocket updates.

Usage:
    from src.orchestrator.llm_routing_litellm import LLMRouterLiteLLM
    from src.orchestrator.providers import CachedAnalysisProvider

    provider = CachedAnalysisProvider()
    router = LLMRouterLiteLLM(provider=provider, model="gemini/gemini-2.0-flash")

    async for event in router.process_question_stream("What's the tempo?", track_id):
        print(event)
"""

import json
from typing import AsyncIterator

import litellm
from asgiref.sync import sync_to_async
from dotenv import load_dotenv

from .base import BaseRouter
from .events import RouterEvent
from .providers import AnalysisProvider
from src.agents.registry import get_tool_schemas_openai
from src.observability.tracing import trace_llm_call, trace_tool_execution

load_dotenv()


class LLMRouterLiteLLM(BaseRouter):
    """
    Async LLM router using LiteLLM for provider abstraction.

    Supports Claude, Gemini, GPT-4, and other providers through LiteLLM.
    Uses the AnalysisProvider strategy for fetching analysis data.
    """

    def __init__(
        self,
        provider: AnalysisProvider,
        model: str = "gemini/gemini-2.0-flash",
        **kwargs,
    ):
        """
        Initialize the LiteLLM router.

        Args:
            provider: Strategy for fetching analysis data
            model: LiteLLM model identifier. Examples:
                - "claude-sonnet-4-20250514" (Anthropic)
                - "gemini/gemini-2.0-flash" (Google)
                - "gpt-4o" (OpenAI)
            **kwargs: Passed to BaseRouter (enable_tracing, project_name)
        """
        super().__init__(provider=provider, **kwargs)
        self.model = model

    def _build_prompt(self, question: str, track_id: str) -> str:
        """Build the system prompt for the LLM."""
        return f"""You have access to audio analysis tools. The user is asking about an audio file.

Track ID: {track_id}

User question: {question}

Use the appropriate analysis tool(s) to answer this question. After receiving the analysis results, provide a clear, natural language response to the user's question."""

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
            RouterEvent dictionaries
        """
        messages = [{"role": "user", "content": self._build_prompt(question, track_id)}]
        tools = get_tool_schemas_openai()

        try:
            with trace_llm_call(self.model, question, tools=[t["function"]["name"] for t in tools]):
                while True:
                    response = await litellm.acompletion(
                        model=self.model,
                        messages=messages,
                        tools=tools,
                    )

                    msg = response.choices[0].message

                    if msg.tool_calls:
                        # Add assistant message to history
                        messages.append(msg.model_dump())

                        for tc in msg.tool_calls:
                            tool_name = tc.function.name
                            agent_name = tool_name.replace("analyse_", "")

                            yield {"type": "tool_call", "tool": tool_name}

                            with trace_tool_execution(agent_name, track_id):
                                try:
                                    result = await sync_to_async(self.provider.get_analysis)(
                                        agent_name, track_id
                                    )
                                    tool_result = {"success": True, "data": result}
                                    yield {"type": "tool_result", "tool": tool_name, "success": True}
                                except Exception as e:
                                    tool_result = {"success": False, "error": str(e)}
                                    yield {"type": "tool_result", "tool": tool_name, "success": False}

                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": json.dumps(tool_result),
                            })

                    else:
                        # No tool calls - use response content directly (no second API call needed)
                        content = msg.content or ""
                        if content:
                            yield {"type": "token", "text": content}
                        yield {"type": "done", "full_response": content}
                        return
        except Exception as e:
            yield {"type": "error", "message": str(e)}
            return
