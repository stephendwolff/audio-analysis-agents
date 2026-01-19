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


def get_tool_schemas_openai() -> list[dict]:
    """
    Generate OpenAI-format tool schemas from registered agents.

    Returns a list of tool definitions compatible with OpenAI/LiteLLM.
    """
    schemas = []
    for cls in AGENT_REGISTRY.values():
        agent = cls()
        schemas.append({
            "type": "function",
            "function": {
                "name": f"analyse_{agent.name}",
                "description": agent.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "track_id": {
                            "type": "string",
                            "description": "Track ID to analyse"
                        }
                    },
                    "required": ["track_id"]
                }
            }
        })
    return schemas
