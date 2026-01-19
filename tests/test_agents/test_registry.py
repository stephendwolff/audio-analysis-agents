# tests/test_agents/test_registry.py
import pytest
from src.agents.registry import (
    AGENT_REGISTRY,
    register_agent,
    get_all_agents,
    get_agent,
    get_tool_schemas_openai,
)


class TestToolSchemaGeneration:
    """Tests for generating tool schemas from registry."""

    def test_get_tool_schemas_openai_returns_list(self):
        """Should return a list of tool schemas."""
        schemas = get_tool_schemas_openai()
        assert isinstance(schemas, list)

    def test_get_tool_schemas_openai_schema_structure(self):
        """Each schema should have correct OpenAI format."""
        schemas = get_tool_schemas_openai()
        assert len(schemas) >= 3

        for schema in schemas:
            assert schema["type"] == "function"
            assert "function" in schema
            func = schema["function"]
            assert "name" in func
            assert func["name"].startswith("analyse_")
            assert "description" in func
            assert len(func["description"]) > 20
            assert "parameters" in func
            params = func["parameters"]
            assert params["type"] == "object"
            assert "track_id" in params["properties"]
            assert "track_id" in params["required"]

    def test_get_tool_schemas_openai_includes_all_agents(self):
        """Should include schemas for all registered agents."""
        schemas = get_tool_schemas_openai()
        schema_names = {s["function"]["name"] for s in schemas}
        assert "analyse_spectral" in schema_names
        assert "analyse_temporal" in schema_names
        assert "analyse_rhythm" in schema_names
