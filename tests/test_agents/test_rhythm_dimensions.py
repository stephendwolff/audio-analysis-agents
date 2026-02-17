"""Tests for the new rhythm dimensions: time_signature, swing, steadiness, upbeat."""

import numpy as np
import pytest
from src.agents.rhythm import RhythmAgent


@pytest.fixture
def agent():
    return RhythmAgent()


@pytest.fixture
def click_track_120bpm():
    """Generate a synthetic click track at 120 BPM, 4/4, straight, steady."""
    sr = 22050
    duration = 4.0
    samples = np.zeros(int(sr * duration), dtype=np.float32)
    beat_interval = 0.5
    for i in range(int(duration / beat_interval)):
        pos = int(i * beat_interval * sr)
        click_len = min(200, len(samples) - pos)
        samples[pos : pos + click_len] = 0.8 * np.sin(
            2 * np.pi * 1000 * np.arange(click_len) / sr
        )
    return samples, sr


class TestNewDimensions:

    def test_result_contains_time_signature(self, agent, click_track_120bpm):
        samples, sr = click_track_120bpm
        result = agent.analyse(samples, sr)
        assert result.success
        assert "time_signature" in result.data
        assert result.data["time_signature"] in ("4/4", "3/4", "other")

    def test_result_contains_swing(self, agent, click_track_120bpm):
        samples, sr = click_track_120bpm
        result = agent.analyse(samples, sr)
        assert result.success
        assert "swing" in result.data
        swing = result.data["swing"]
        assert isinstance(swing, float)
        assert 0.0 <= swing <= 1.0

    def test_result_contains_steadiness(self, agent, click_track_120bpm):
        samples, sr = click_track_120bpm
        result = agent.analyse(samples, sr)
        assert result.success
        assert "steadiness" in result.data
        steadiness = result.data["steadiness"]
        assert isinstance(steadiness, float)
        assert 0.0 <= steadiness <= 1.0

    def test_result_contains_upbeat(self, agent, click_track_120bpm):
        samples, sr = click_track_120bpm
        result = agent.analyse(samples, sr)
        assert result.success
        assert "upbeat" in result.data
        assert isinstance(result.data["upbeat"], bool)

    def test_beat_times_not_truncated(self, agent):
        """beat_times should include all beats, not just first 20."""
        sr = 22050
        duration = 30.0
        samples = np.zeros(int(sr * duration), dtype=np.float32)
        beat_interval = 0.5
        for i in range(int(duration / beat_interval)):
            pos = int(i * beat_interval * sr)
            click_len = min(200, len(samples) - pos)
            samples[pos : pos + click_len] = 0.8 * np.sin(
                2 * np.pi * 1000 * np.arange(click_len) / sr
            )
        result = agent.analyse(samples, sr)
        assert result.success
        assert len(result.data["beat_times"]) > 20

    def test_onset_times_not_truncated(self, agent):
        """onset_times should include all onsets, not just first 30."""
        sr = 22050
        duration = 30.0
        samples = np.zeros(int(sr * duration), dtype=np.float32)
        beat_interval = 0.25
        for i in range(int(duration / beat_interval)):
            pos = int(i * beat_interval * sr)
            click_len = min(100, len(samples) - pos)
            samples[pos : pos + click_len] = 0.8 * np.sin(
                2 * np.pi * 2000 * np.arange(click_len) / sr
            )
        result = agent.analyse(samples, sr)
        assert result.success
        assert len(result.data["onset_times"]) > 30

    def test_steady_click_has_high_steadiness(self, agent, click_track_120bpm):
        samples, sr = click_track_120bpm
        result = agent.analyse(samples, sr)
        assert result.success
        assert result.data["steadiness"] >= 0.5

    def test_straight_click_has_low_swing(self, agent, click_track_120bpm):
        samples, sr = click_track_120bpm
        result = agent.analyse(samples, sr)
        assert result.success
        assert result.data["swing"] <= 0.5
