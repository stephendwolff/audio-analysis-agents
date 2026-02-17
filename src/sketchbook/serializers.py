"""Maps agent output to the Musical Sketchbook API contract response shape."""

from src.agents.base import AnalysisResult
from .descriptors import generate_descriptors


def build_analysis_result(
    rhythm: AnalysisResult,
    spectral: AnalysisResult,
    temporal: AnalysisResult,
) -> dict:
    """
    Transform agent results into the contract response shape.

    Args:
        rhythm: Result from RhythmAgent
        spectral: Result from SpectralAgent
        temporal: Result from TemporalAgent

    Returns:
        Dict matching the /analyse response contract
    """
    dimensions = {
        "bpm": rhythm.data["tempo_bpm"],
        "time_signature": rhythm.data["time_signature"],
        "swing": rhythm.data["swing"],
        "steadiness": rhythm.data["steadiness"],
        "upbeat": rhythm.data["upbeat"],
    }

    return {
        "dimensions": dimensions,
        "descriptors": generate_descriptors(dimensions),
        "raw_data": {
            "beats": rhythm.data["beat_times"],
            "onsets": rhythm.data["onset_times"],
            "spectral_centroid_mean": spectral.data["spectral_centroid"]["mean"],
            "rms_energy_mean": temporal.data["rms_energy"]["mean"],
        },
    }
