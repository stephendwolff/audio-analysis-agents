"""
Orchestrator module for LLM routing.

Provides routers that connect user questions to analysis tools via LLMs.
"""

from .orchestrator import Orchestrator
from .llm_routing import LLMRouter
from .events import (
    RouterEvent,
    ToolCallEvent,
    ToolResultEvent,
    TokenEvent,
    DoneEvent,
    ErrorEvent,
)
from .providers import (
    AnalysisProvider,
    CachedAnalysisProvider,
)
from .base import BaseRouter
from .llm_routing_litellm import LLMRouterLiteLLM

__all__ = [
    # Legacy
    "Orchestrator",
    "LLMRouter",
    # Events
    "RouterEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "TokenEvent",
    "DoneEvent",
    "ErrorEvent",
    # Providers
    "AnalysisProvider",
    "CachedAnalysisProvider",
    # Routers
    "BaseRouter",
    "LLMRouterLiteLLM",
]

# LangGraph version - import separately to avoid requiring langgraph dependency
try:
    from .llm_routing_langgraph import LLMRouterLangGraph
    __all__.append("LLMRouterLangGraph")
except ImportError:
    pass
