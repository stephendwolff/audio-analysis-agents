"""Tests for the orchestrator."""

import numpy as np
import pytest

from src.orchestrator import Orchestrator
from src.agents import SpectralAgent, TemporalAgent, RhythmAgent


class TestOrchestrator:
    """Tests for orchestrator functionality."""

    def test_init_registers_default_agents(self):
        """Orchestrator should register default agents on init."""
        orch = Orchestrator()
        agents = orch.list_agents()

        assert "spectral" in agents
        assert "temporal" in agents
        assert "rhythm" in agents

    def test_register_custom_agent(self):
        """Should be able to register custom agents."""
        from src.agents.base import BaseAgent, AnalysisResult

        class CustomAgent(BaseAgent):
            name = "custom"
            description = "Test agent"

            def analyse(self, samples, sample_rate):
                return AnalysisResult(agent=self.name, success=True, data={"test": 1})

        orch = Orchestrator()
        orch.register_agent(CustomAgent())

        assert "custom" in orch.list_agents()


class TestAgentsWithSyntheticData:
    """Test agents with synthetic audio data."""

    @pytest.fixture
    def sine_wave(self):
        """Generate a 440Hz sine wave."""
        sr = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        samples = 0.5 * np.sin(2 * np.pi * 440 * t)
        return samples, sr

    @pytest.fixture
    def click_track(self):
        """Generate a simple click track at 120 BPM."""
        sr = 22050
        duration = 4.0
        samples = np.zeros(int(sr * duration))

        # Add clicks at 120 BPM (0.5s intervals)
        click_interval = int(0.5 * sr)
        for i in range(0, len(samples), click_interval):
            # Short click
            click_len = min(100, len(samples) - i)
            samples[i : i + click_len] = 0.8

        return samples, sr

    def test_spectral_agent_with_sine(self, sine_wave):
        """Spectral agent should detect 440Hz as dominant frequency."""
        samples, sr = sine_wave
        agent = SpectralAgent()

        result = agent.analyse(samples, sr)

        assert result.success
        assert "dominant_frequencies" in result.data

        # Check that 440Hz is among dominant frequencies
        dom_freqs = [f["frequency_hz"] for f in result.data["dominant_frequencies"]]
        assert any(abs(f - 440) < 10 for f in dom_freqs)

    def test_temporal_agent_with_sine(self, sine_wave):
        """Temporal agent should return duration and amplitude stats."""
        samples, sr = sine_wave
        agent = TemporalAgent()

        result = agent.analyse(samples, sr)

        assert result.success
        assert result.data["duration_seconds"] == pytest.approx(1.0, rel=0.01)
        assert "amplitude" in result.data
        assert "rms_energy" in result.data

    def test_rhythm_agent_with_clicks(self, click_track):
        """Rhythm agent should detect ~120 BPM from click track."""
        samples, sr = click_track
        agent = RhythmAgent()

        result = agent.analyse(samples, sr)

        assert result.success
        assert "tempo_bpm" in result.data
        # Allow some tolerance in BPM detection
        assert 100 < result.data["tempo_bpm"] < 140
