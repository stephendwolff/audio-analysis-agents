import pytest
from src.agents import SpectralAgent, TemporalAgent, RhythmAgent


@pytest.mark.parametrize("agent_class,expected_name", [
    (SpectralAgent, "spectral"),
    (TemporalAgent, "temporal"),
    (RhythmAgent, "rhythm"),
])
def test_agent_has_name_and_description(agent_class, expected_name):
    """All agents must have name and description."""
    agent = agent_class()
    assert agent.name == expected_name
    assert isinstance(agent.description, str)
    assert len(agent.description) > 20  # Meaningful description
