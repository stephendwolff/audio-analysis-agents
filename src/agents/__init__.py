"""Specialised analysis agents."""

from .base import BaseAgent, AnalysisResult
from .registry import register_agent, get_all_agents, get_agent, AGENT_REGISTRY, get_tool_schemas_openai
from .spectral import SpectralAgent
from .temporal import TemporalAgent
from .rhythm import RhythmAgent

__all__ = [
    "BaseAgent",
    "AnalysisResult",
    "register_agent",
    "get_all_agents",
    "get_agent",
    "get_tool_schemas_openai",
    "AGENT_REGISTRY",
    "SpectralAgent",
    "TemporalAgent",
    "RhythmAgent",
]
