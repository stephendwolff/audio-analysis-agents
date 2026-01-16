"""
LLM Routing Module - LangGraph Implementation

This module provides the same functionality as llm_routing.py but uses
LangGraph's StateGraph abstraction. Compare this with the raw implementation
to see what LangGraph handles for you.

Key LangGraph concepts:
- StateGraph: Defines conversation flow as nodes and edges
- State: TypedDict holding conversation state (messages, results)
- Nodes: Functions that process state and return updates
- Conditional edges: Route based on LLM decisions
- ToolNode: Built-in node that handles tool execution

Usage:
    from src.orchestrator.llm_routing_langgraph import LLMRouterLangGraph

    router = LLMRouterLangGraph()
    response = router.process_question("What's the tempo?", audio_path="track.wav")
"""

import os
from pathlib import Path
from typing import Any, Annotated, Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from ..agents import SpectralAgent, TemporalAgent, RhythmAgent
from ..tools.loader import load_audio, AudioData
from ..observability.tracing import init_tracing, trace_tool_execution

load_dotenv()


# ---------------------------------------------------------------------------
# State Definition
# ---------------------------------------------------------------------------
# LangGraph uses a TypedDict to define what data flows through the graph.
# The `add_messages` annotation tells LangGraph to append messages rather
# than replace them.

class AgentState(TypedDict):
    """State that flows through the graph."""
    messages: Annotated[list, add_messages]  # Conversation history
    audio_path: str                           # Path to audio file being analysed
    tools_called: list[str]                   # Track which tools were invoked
    analyses: dict[str, Any]                  # Store analysis results


# ---------------------------------------------------------------------------
# Tool Definitions
# ---------------------------------------------------------------------------
# LangGraph uses @tool decorated functions. The docstring becomes the
# tool description that the LLM sees.

# We need a way to pass audio_path to tools. LangGraph tools receive
# the arguments from the LLM, so we'll use a module-level cache.
_audio_cache: dict[str, AudioData] = {}
_current_audio_path: str = ""


def _load_audio_cached(audio_path: str) -> AudioData:
    """Load audio with caching."""
    if audio_path not in _audio_cache:
        _audio_cache[audio_path] = load_audio(audio_path, target_sr=22050, mono=True)
    return _audio_cache[audio_path]


@tool
def analyse_spectral(track_id: str) -> dict:
    """Analyse frequency content of audio. Returns spectral centroid (brightness), bandwidth, rolloff, flatness (tonal vs noisy), dominant frequencies, and MFCC summary. Use for questions about frequency, pitch, tone, timbre, or 'what does it sound like'."""
    # Use the current audio path if track_id doesn't exist as a file
    audio_path = track_id if Path(track_id).exists() else _current_audio_path

    with trace_tool_execution("spectral", audio_path):
        try:
            audio = _load_audio_cached(audio_path)
            agent = SpectralAgent()
            result = agent.analyse(audio.samples, audio.sample_rate)
            return {"success": result.success, "data": result.data, "error": result.error}
        except Exception as e:
            return {"success": False, "error": str(e)}


@tool
def analyse_temporal(track_id: str) -> dict:
    """Analyse time-domain properties of audio. Returns duration, RMS energy (loudness), peak amplitude, dynamic range, zero crossing rate, and amplitude envelope. Use for questions about duration, volume, dynamics, loudness, or energy."""
    audio_path = track_id if Path(track_id).exists() else _current_audio_path

    with trace_tool_execution("temporal", audio_path):
        try:
            audio = _load_audio_cached(audio_path)
            agent = TemporalAgent()
            result = agent.analyse(audio.samples, audio.sample_rate)
            return {"success": result.success, "data": result.data, "error": result.error}
        except Exception as e:
            return {"success": False, "error": str(e)}


@tool
def analyse_rhythm(track_id: str) -> dict:
    """Analyse tempo and rhythmic properties of audio. Returns estimated BPM, beat positions, onset times, and tempo stability. Use for questions about tempo, BPM, beats, rhythm, or timing."""
    audio_path = track_id if Path(track_id).exists() else _current_audio_path

    with trace_tool_execution("rhythm", audio_path):
        try:
            audio = _load_audio_cached(audio_path)
            agent = RhythmAgent()
            result = agent.analyse(audio.samples, audio.sample_rate)
            return {"success": result.success, "data": result.data, "error": result.error}
        except Exception as e:
            return {"success": False, "error": str(e)}


# Bundle all tools
TOOLS = [analyse_spectral, analyse_temporal, analyse_rhythm]

# Map tool names to agent names for tracking
TOOL_NAME_TO_AGENT = {
    "analyse_spectral": "spectral",
    "analyse_temporal": "temporal",
    "analyse_rhythm": "rhythm",
}


# ---------------------------------------------------------------------------
# Graph Nodes
# ---------------------------------------------------------------------------

def call_model(state: AgentState) -> dict:
    """
    Node that calls the LLM.

    Takes the current messages and gets a response from Gemini.
    The LLM may return tool calls or a final text response.
    """
    # Create model with tools bound
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    ).bind_tools(TOOLS)

    # Call the model
    response = model.invoke(state["messages"])

    # Return the response to be added to messages
    return {"messages": [response]}


def process_tool_results(state: AgentState) -> dict:
    """
    Node that processes tool results after ToolNode executes them.

    Updates the tools_called and analyses tracking in state.
    """
    # Find the most recent tool messages
    tools_called = list(state.get("tools_called", []))
    analyses = dict(state.get("analyses", {}))

    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage):
            tool_name = msg.name
            if tool_name in TOOL_NAME_TO_AGENT:
                agent_name = TOOL_NAME_TO_AGENT[tool_name]
                if f"analyse_{agent_name}" not in tools_called:
                    tools_called.append(f"analyse_{agent_name}")
                    # Parse the tool result
                    try:
                        import json
                        result = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                        analyses[f"analyse_{agent_name}"] = result
                    except:
                        analyses[f"analyse_{agent_name}"] = {"raw": msg.content}
        elif isinstance(msg, AIMessage):
            # Stop when we hit the AI message that triggered these tools
            break

    return {"tools_called": tools_called, "analyses": analyses}


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """
    Conditional edge that decides whether to call tools or end.

    Checks the last message - if it has tool calls, route to tools.
    Otherwise, we're done.
    """
    last_message = state["messages"][-1]

    # Check if the LLM wants to call tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return "end"


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """
    Build the LangGraph StateGraph.

    Graph structure:
        ┌─────────────┐
        │   START     │
        └──────┬──────┘
               │
               ▼
        ┌─────────────┐
        │ call_model  │◄─────────────┐
        └──────┬──────┘              │
               │                     │
               ▼                     │
        ┌─────────────┐              │
        │should_continue│            │
        └──────┬──────┘              │
               │                     │
       ┌───────┴───────┐             │
       │               │             │
       ▼               ▼             │
    ┌─────┐      ┌──────────┐        │
    │ END │      │  tools   │────────┘
    └─────┘      │(ToolNode)│
                 └──────────┘
    """
    # Create the graph with our state schema
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("call_model", call_model)
    graph.add_node("tools", ToolNode(tools=TOOLS))
    graph.add_node("process_results", process_tool_results)

    # Set entry point
    graph.set_entry_point("call_model")

    # Add conditional edge from call_model
    graph.add_conditional_edges(
        "call_model",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        }
    )

    # After tools execute, process results then go back to call_model
    graph.add_edge("tools", "process_results")
    graph.add_edge("process_results", "call_model")

    return graph.compile()


# ---------------------------------------------------------------------------
# Router Class
# ---------------------------------------------------------------------------

class LLMRouterLangGraph:
    """
    Routes questions through Gemini LLM to appropriate analysis agents.

    This is the LangGraph implementation - compare with LLMRouter to see
    what the framework handles for you.
    """

    def __init__(
        self,
        enable_tracing: bool = True,
        project_name: str = "audio-analysis-agents",
    ):
        """
        Initialise the LangGraph router.

        Args:
            enable_tracing: Whether to enable Opik tracing
            project_name: Project name for Opik
        """
        self.enable_tracing = enable_tracing

        # Check for API key
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        # Initialise tracing
        if enable_tracing:
            init_tracing(project_name)

        # Build the graph
        self.graph = build_graph()

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
        global _current_audio_path
        audio_path = str(audio_path)
        _current_audio_path = audio_path

        # Build initial message
        prompt = f"""You have access to audio analysis tools. The user is asking about an audio file.

Audio file: {audio_path}

User question: {question}

Use the appropriate analysis tool(s) to answer this question. After receiving the analysis results, provide a clear, natural language response to the user's question."""

        if verbose:
            print(f"\n{'='*60}")
            print(f"Question: {question}")
            print(f"Audio: {audio_path}")
            print(f"{'='*60}")

        # Initial state
        initial_state = {
            "messages": [HumanMessage(content=prompt)],
            "audio_path": audio_path,
            "tools_called": [],
            "analyses": {},
        }

        # Run the graph
        final_state = self.graph.invoke(initial_state)

        # Extract the final response
        last_message = final_state["messages"][-1]
        response_text = last_message.content if hasattr(last_message, "content") else str(last_message)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Response: {response_text}")
            print(f"{'='*60}")

        return {
            "response": response_text,
            "tools_called": final_state.get("tools_called", []),
            "analyses": final_state.get("analyses", {}),
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

        # Create model with tools
        model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        ).bind_tools(TOOLS)

        # Get response
        response = model.invoke([HumanMessage(content=prompt)])

        # Extract tool names
        selected = []
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                agent_name = tool_name.replace("analyse_", "")
                selected.append(agent_name)

        return selected
