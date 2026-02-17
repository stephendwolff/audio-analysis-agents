"""Tests for the response builder that maps agent output to contract shape."""

import pytest
from src.agents.base import AnalysisResult
from src.sketchbook.serializers import build_analysis_result


@pytest.fixture
def rhythm_result():
    return AnalysisResult(
        agent="rhythm",
        success=True,
        data={
            "tempo_bpm": 120.5,
            "time_signature": "4/4",
            "swing": 0.15,
            "steadiness": 0.87,
            "upbeat": False,
            "beat_times": [0.5, 1.0, 1.5, 2.0],
            "onset_times": [0.1, 0.5, 0.8, 1.0, 1.5],
        },
    )


@pytest.fixture
def spectral_result():
    return AnalysisResult(
        agent="spectral",
        success=True,
        data={"spectral_centroid": {"mean": 2400.5, "std": 100}},
    )


@pytest.fixture
def temporal_result():
    return AnalysisResult(
        agent="temporal",
        success=True,
        data={"rms_energy": {"mean": 0.034, "std": 0.01}},
    )


class TestBuildAnalysisResult:

    def test_has_dimensions(self, rhythm_result, spectral_result, temporal_result):
        result = build_analysis_result(rhythm_result, spectral_result, temporal_result)
        dims = result["dimensions"]
        assert dims["bpm"] == 120.5
        assert dims["time_signature"] == "4/4"
        assert dims["swing"] == 0.15
        assert dims["steadiness"] == 0.87
        assert dims["upbeat"] is False

    def test_has_descriptors(self, rhythm_result, spectral_result, temporal_result):
        result = build_analysis_result(rhythm_result, spectral_result, temporal_result)
        assert isinstance(result["descriptors"], list)
        assert len(result["descriptors"]) > 0

    def test_has_raw_data(self, rhythm_result, spectral_result, temporal_result):
        result = build_analysis_result(rhythm_result, spectral_result, temporal_result)
        raw = result["raw_data"]
        assert raw["beats"] == [0.5, 1.0, 1.5, 2.0]
        assert raw["onsets"] == [0.1, 0.5, 0.8, 1.0, 1.5]
        assert raw["spectral_centroid_mean"] == 2400.5
        assert raw["rms_energy_mean"] == 0.034
