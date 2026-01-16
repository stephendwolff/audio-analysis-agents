"""
LLM Routing Module - Gemini Function Calling with Real Agents

This module connects the Gemini LLM to the actual audio analysis agents.
It handles:
- Tool schema definitions matching the agent interfaces
- The multi-turn Gemini ↔ agent conversation loop
- Opik tracing for observability

Usage:
    from src.orchestrator.llm_routing import LLMRouter

    router = LLMRouter()
    response = router.process_question("What's the tempo?", audio_path="track.wav")
"""

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from ..agents import SpectralAgent, TemporalAgent, RhythmAgent, AnalysisResult
from ..tools.loader import load_audio, AudioData
from ..observability.tracing import init_tracing, trace_llm_call, trace_tool_execution

load_dotenv()


# ---------------------------------------------------------------------------
# Tool Definitions - Match the real agent interfaces
# ---------------------------------------------------------------------------

TOOL_DECLARATIONS = [
    types.FunctionDeclaration(
        name="analyse_spectral",
        description=(
            "Analyse frequency content of audio. Returns spectral centroid (brightness), "
            "bandwidth, rolloff, flatness (tonal vs noisy), dominant frequencies, and MFCC summary. "
            "Use for questions about frequency, pitch, tone, timbre, or 'what does it sound like'."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "track_id": types.Schema(
                    type=types.Type.STRING,
                    description="ID or path of the audio track to analyse"
                )
            },
            required=["track_id"]
        )
    ),
    types.FunctionDeclaration(
        name="analyse_temporal",
        description=(
            "Analyse time-domain properties of audio. Returns duration, RMS energy (loudness), "
            "peak amplitude, dynamic range, zero crossing rate, and amplitude envelope. "
            "Use for questions about duration, volume, dynamics, loudness, or energy."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "track_id": types.Schema(
                    type=types.Type.STRING,
                    description="ID or path of the audio track to analyse"
                )
            },
            required=["track_id"]
        )
    ),
    types.FunctionDeclaration(
        name="analyse_rhythm",
        description=(
            "Analyse tempo and rhythmic properties of audio. Returns estimated BPM, "
            "beat positions, onset times, and tempo stability. "
            "Use for questions about tempo, BPM, beats, rhythm, or timing."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "track_id": types.Schema(
                    type=types.Type.STRING,
                    description="ID or path of the audio track to analyse"
                )
            },
            required=["track_id"]
        )
    ),
]

AUDIO_TOOLS = types.Tool(function_declarations=TOOL_DECLARATIONS)

# Map tool names to agent classes
TOOL_TO_AGENT = {
    "analyse_spectral": SpectralAgent,
    "analyse_temporal": TemporalAgent,
    "analyse_rhythm": RhythmAgent,
}


class LLMRouter:
    """
    Routes questions through Gemini LLM to appropriate analysis agents.

    The router:
    1. Takes a user question and audio file path
    2. Sends the question to Gemini with tool definitions
    3. When Gemini requests a tool call, executes the corresponding agent
    4. Sends results back to Gemini for natural language response
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        enable_tracing: bool = True,
        project_name: str = "audio-analysis-agents",
    ):
        """
        Initialise the LLM router.

        Args:
            model: Gemini model to use
            enable_tracing: Whether to enable Opik tracing
            project_name: Project name for Opik
        """
        self.model = model
        self.enable_tracing = enable_tracing

        # Initialise Gemini client
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        self.client = genai.Client(api_key=api_key)

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
        """
        Execute an analysis tool/agent.

        Args:
            tool_name: Name of the tool (e.g., "analyse_spectral")
            track_id: Path to the audio file

        Returns:
            Analysis result as a dictionary
        """
        if tool_name not in self._agents:
            return {"error": f"Unknown tool: {tool_name}"}

        agent = self._agents[tool_name]

        with trace_tool_execution(agent.name, track_id):
            try:
                # Load audio
                audio = self._load_audio(track_id)

                # Run analysis
                result: AnalysisResult = agent.analyse(audio.samples, audio.sample_rate)

                if result.success:
                    return {"success": True, "data": result.data}
                else:
                    return {"success": False, "error": result.error}

            except FileNotFoundError as e:
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

        # Build initial prompt with context
        prompt = f"""You have access to audio analysis tools. The user is asking about an audio file.

Audio file: {audio_path}

User question: {question}

Use the appropriate analysis tool(s) to answer this question. After receiving the analysis results, provide a clear, natural language response to the user's question."""

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)]
            )
        ]

        tools_called = []
        analyses = {}

        if verbose:
            print(f"\n{'='*60}")
            print(f"Question: {question}")
            print(f"Audio: {audio_path}")
            print(f"{'='*60}")

        # Conversation loop
        with trace_llm_call(self.model, prompt, tools=list(TOOL_TO_AGENT.keys())):
            while True:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=types.GenerateContentConfig(tools=[AUDIO_TOOLS])
                )

                candidate = response.candidates[0]
                parts = candidate.content.parts

                # Check for function calls
                function_calls = []
                text_response = None

                for part in parts:
                    if part.function_call:
                        function_calls.append(part.function_call)
                    elif part.text:
                        text_response = part.text

                if function_calls:
                    # Add assistant response to history
                    contents.append(candidate.content)

                    # Execute each tool
                    function_responses = []
                    for func_call in function_calls:
                        func_name = func_call.name
                        func_args = dict(func_call.args)
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

                        function_responses.append(
                            types.Part.from_function_response(
                                name=func_name,
                                response={"result": result}
                            )
                        )

                    # Add function responses to history
                    contents.append(
                        types.Content(
                            role="user",
                            parts=function_responses
                        )
                    )

                elif text_response:
                    if verbose:
                        print(f"\n{'='*60}")
                        print(f"Response: {text_response}")
                        print(f"{'='*60}")

                    return {
                        "response": text_response,
                        "tools_called": tools_called,
                        "analyses": analyses,
                    }

                else:
                    return {
                        "response": "No response generated",
                        "tools_called": tools_called,
                        "analyses": analyses,
                        "error": "LLM returned neither function calls nor text",
                    }

    def get_tool_selection(self, question: str, track_id: str) -> list[str]:
        """
        Get which tools the LLM would select for a question (without executing them).

        Useful for evaluation.

        Args:
            question: The user's question
            track_id: Track identifier for context

        Returns:
            List of agent names (e.g., ["spectral", "rhythm"])
        """
        prompt = f"""You have access to audio analysis tools. The user is asking about audio track '{track_id}'.

User question: {question}

Use the appropriate analysis tool(s) to answer this question."""

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)]
            )
        ]

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(tools=[AUDIO_TOOLS])
        )

        # Extract tool names
        selected = []
        candidate = response.candidates[0]

        for part in candidate.content.parts:
            if part.function_call:
                func_name = part.function_call.name
                # Convert tool name to agent name
                agent_name = func_name.replace("analyse_", "")
                selected.append(agent_name)

        return selected
