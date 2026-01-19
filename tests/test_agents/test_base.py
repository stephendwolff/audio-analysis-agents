# tests/test_agents/test_base.py
import pytest
from src.agents.base import BaseAgent, AnalysisResult


def test_base_agent_requires_description():
    """BaseAgent subclasses must define description."""

    class IncompleteAgent(BaseAgent):
        name = "incomplete"
        # Missing description

        def analyse(self, samples, sample_rate):
            return AnalysisResult(agent=self.name, success=True, data={})

    with pytest.raises(TypeError, match="description"):
        IncompleteAgent()


def test_base_agent_with_description_instantiates():
    """BaseAgent subclasses with description can be instantiated."""

    class CompleteAgent(BaseAgent):
        name = "complete"
        description = "A complete test agent"

        def analyse(self, samples, sample_rate):
            return AnalysisResult(agent=self.name, success=True, data={})

    agent = CompleteAgent()
    assert agent.name == "complete"
    assert agent.description == "A complete test agent"
