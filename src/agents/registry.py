"""Agent registry for automatic registration and discovery."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseAgent

AGENT_REGISTRY: dict[str, type["BaseAgent"]] = {}


def register_agent(cls):
    """Decorator to register an agent for automatic analysis."""
    AGENT_REGISTRY[cls.name] = cls
    return cls


def get_all_agents() -> list["BaseAgent"]:
    """Return instances of all registered agents."""
    return [cls() for cls in AGENT_REGISTRY.values()]


def get_agent(name: str) -> "BaseAgent | None":
    """Get a specific agent by name."""
    cls = AGENT_REGISTRY.get(name)
    return cls() if cls else None
