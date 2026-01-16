"""
LLM Routing Module - LiteLLM Provider-Agnostic Implementation

This module provides the same functionality as llm_routing.py but uses
LiteLLM for provider abstraction. The same code works with:
- Claude (Anthropic)
- Gemini (Google)
- GPT-4 (OpenAI)
- And many others

LiteLLM uses OpenAI's tool calling format, which it translates to each
provider's native format.

Usage:
    from src.orchestrator.llm_routing_litellm import LLMRouterLiteLLM

    # Use Claude
    router = LLMRouterLiteLLM(model="claude-sonnet-4-20250514")

    # Use Gemini
    router = LLMRouterLiteLLM(model="gemini/gemini-2.0-flash")

    # Use GPT-4
    router = LLMRouterLiteLLM(model="gpt-4o")

    response = router.process_question("What's the tempo?", audio_path="track.wav")

Environment variables needed:
- ANTHROPIC_API_KEY for Claude
- GOOGLE_API_KEY for Gemini (note: LiteLLM uses GEMINI_API_KEY or GOOGLE_API_KEY)
- OPENAI_API_KEY for GPT-4
"""

import json
import os
from pathlib import Path
from typing import Any

import litellm
from dotenv import load_dotenv

from ..agents import SpectralAgent, TemporalAgent, RhythmAgent
from ..tools.loader import load_audio, AudioData
from ..observability.tracing import init_tracing, trace_llm_call, trace_tool_execution

load_dotenv()


# ---------------------------------------------------------------------------
# Tool Definitions - OpenAI format (LiteLLM standard)
# ---------------------------------------------------------------------------
# LiteLLM uses OpenAI's tool format, which it translates to each provider's
# native format automatically.

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "analyse_spectral",
            "description": (
                "Analyse frequency content of audio. Returns spectral centroid (brightness), "
                "bandwidth, rolloff, flatness (tonal vs noisy), dominant frequencies, and MFCC summary. "
                "Use for questions about frequency, pitch, tone, timbre, or 'what does it sound like'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "track_id": {
                        "type": "string",
                        "description": "ID or path of the audio track to analyse"
                    }
                },
                "required": ["track_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyse_temporal",
            "description": (
                "Analyse time-domain properties of audio. Returns duration, RMS energy (loudness), "
                "peak amplitude, dynamic range, zero crossing rate, and amplitude envelope. "
                "Use for questions about duration, volume, dynamics, loudness, or energy."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "track_id": {
                        "type": "string",
                        "description": "ID or path of the audio track to analyse"
                    }
                },
                "required": ["track_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyse_rhythm",
            "description": (
                "Analyse tempo and rhythmic properties of audio. Returns estimated BPM, "
                "beat positions, onset times, and tempo stability. "
                "Use for questions about tempo, BPM, beats, rhythm, or timing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "track_id": {
                        "type": "string",
                        "description": "ID or path of the audio track to analyse"
                    }
                },
                "required": ["track_id"]
            }
        }
    },
]

# Map tool names to agent classes
TOOL_TO_AGENT = {
    "analyse_spectral": SpectralAgent,
    "analyse_temporal": TemporalAgent,
    "analyse_rhythm": RhythmAgent,
}


class LLMRouterLiteLLM:
    """
    Routes questions through LLM to appropriate analysis agents.

    Uses LiteLLM for provider abstraction - same code works with
    Claude, Gemini, GPT-4, and many others.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        enable_tracing: bool = True,
        project_name: str = "audio-analysis-agents",
    ):
        """
        Initialise the LiteLLM router.

        Args:
            model: Model identifier. Examples:
                - "claude-sonnet-4-20250514" (Anthropic)
                - "gemini/gemini-2.0-flash" (Google)
                - "gpt-4o" (OpenAI)
            enable_tracing: Whether to enable Opik tracing
            project_name: Project name for Opik
        """
        self.model = model
        self.enable_tracing = enable_tracing

        # Initialise tracing
        if enable_tracing:
            init_tracing(project_name)

        # Cache for loaded audio
        self._audio_cache: dict[str, AudioData] = {}

        # Instantiate agents
        self._agents = {
            name: agent_class()
            for name, agent_class in TOOL_TO_AGENT.items()
        }

    def _load_audio(self, audio_path: str | Path) -> AudioData:
        """Load audio file, using cache if available."""
        path_str = str(audio_path)
        if path_str not in self._audio_cache:
            self._audio_cache[path_str] = load_audio(audio_path, target_sr=22050, mono=True)
        return self._audio_cache[path_str]

    def _execute_tool(self, tool_name: str, track_id: str) -> dict[str, Any]:
        """Execute an analysis tool/agent."""
        if tool_name not in self._agents:
            return {"error": f"Unknown tool: {tool_name}"}

        agent = self._agents[tool_name]

        with trace_tool_execution(agent.name, track_id):
            try:
                audio = self._load_audio(track_id)
                result = agent.analyse(audio.samples, audio.sample_rate)

                if result.success:
                    return {"success": True, "data": result.data}
                else:
                    return {"success": False, "error": result.error}

            except FileNotFoundError:
                return {"success": False, "error": f"Audio file not found: {track_id}"}
            except Exception as e:
                return {"success": False, "error": str(e)}

    def process_question(
        self,
        question: str,
        audio_path: str | Path,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """
        Process a user question about an audio file.

        Args:
            question: The user's question
            audio_path: Path to the audio file
            verbose: Print debug information

        Returns:
            Dict with 'response' (text), 'tools_called', and 'analyses'
        """
        audio_path = str(audio_path)

        # Build initial prompt
        prompt = f"""You have access to audio analysis tools. The user is asking about an audio file.

Audio file: {audio_path}

User question: {question}

Use the appropriate analysis tool(s) to answer this question. After receiving the analysis results, provide a clear, natural language response to the user's question."""

        # Build messages in OpenAI format
        messages = [{"role": "user", "content": prompt}]

        tools_called = []
        analyses = {}

        if verbose:
            print(f"\n{'='*60}")
            print(f"Question: {question}")
            print(f"Audio: {audio_path}")
            print(f"Model: {self.model}")
            print(f"{'='*60}")

        # Conversation loop
        with trace_llm_call(self.model, prompt, tools=list(TOOL_TO_AGENT.keys())):
            while True:
                # Call LLM via LiteLLM
                response = litellm.completion(
                    model=self.model,
                    messages=messages,
                    tools=TOOLS,
                )

                # Get the assistant's message
                assistant_message = response.choices[0].message

                # Check for tool calls
                if assistant_message.tool_calls:
                    # Add assistant message to history
                    messages.append(assistant_message.model_dump())

                    # Process each tool call
                    for tool_call in assistant_message.tool_calls:
                        func_name = tool_call.function.name
                        func_args = json.loads(tool_call.function.arguments)
                        track_id = func_args.get("track_id", audio_path)

                        if verbose:
                            print(f"\n[Tool call: {func_name}]")

                        # Execute the tool
                        result = self._execute_tool(func_name, track_id)

                        tools_called.append(func_name)
                        analyses[func_name] = result

                        if verbose:
                            success = result.get("success", False)
                            print(f"  Success: {success}")

                        # Add tool response to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result),
                        })

                # Check for final text response
                elif assistant_message.content:
                    if verbose:
                        print(f"\n{'='*60}")
                        print(f"Response: {assistant_message.content}")
                        print(f"{'='*60}")

                    return {
                        "response": assistant_message.content,
                        "tools_called": tools_called,
                        "analyses": analyses,
                    }

                else:
                    return {
                        "response": "No response generated",
                        "tools_called": tools_called,
                        "analyses": analyses,
                        "error": "LLM returned neither tool calls nor text",
                    }

    def get_tool_selection(self, question: str, track_id: str) -> list[str]:
        """
        Get which tools the LLM would select for a question.

        For evaluation - matches the interface of LLMRouter.

        Args:
            question: The user's question
            track_id: Track identifier for context

        Returns:
            List of agent names (e.g., ["spectral", "rhythm"])
        """
        prompt = f"""You have access to audio analysis tools. The user is asking about audio track '{track_id}'.

User question: {question}

Use the appropriate analysis tool(s) to answer this question."""

        messages = [{"role": "user", "content": prompt}]

        response = litellm.completion(
            model=self.model,
            messages=messages,
            tools=TOOLS,
        )

        # Extract tool names
        selected = []
        assistant_message = response.choices[0].message

        if assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                func_name = tool_call.function.name
                agent_name = func_name.replace("analyse_", "")
                selected.append(agent_name)

        return selected
