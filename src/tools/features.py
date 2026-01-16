"""Feature extraction helpers."""

from typing import Any

import numpy as np


def extract_features(
    samples: np.ndarray,
    sample_rate: int,
    features: list[str] | None = None,
) -> dict[str, Any]:
    """
    Extract common audio features.

    Args:
        samples: Audio samples as numpy array
        sample_rate: Sample rate in Hz
        features: List of features to extract (None for all)

    Returns:
        Dictionary of feature names to values
    """
    import librosa

    available_features = features or [
        "rms",
        "zcr",
        "spectral_centroid",
        "spectral_bandwidth",
    ]

    results = {}

    if "rms" in available_features:
        rms = librosa.feature.rms(y=samples)[0]
        results["rms_mean"] = float(np.mean(rms))
        results["rms_std"] = float(np.std(rms))

    if "zcr" in available_features:
        zcr = librosa.feature.zero_crossing_rate(samples)[0]
        results["zcr_mean"] = float(np.mean(zcr))
        results["zcr_std"] = float(np.std(zcr))

    if "spectral_centroid" in available_features:
        centroid = librosa.feature.spectral_centroid(y=samples, sr=sample_rate)[0]
        results["spectral_centroid_mean"] = float(np.mean(centroid))
        results["spectral_centroid_std"] = float(np.std(centroid))

    if "spectral_bandwidth" in available_features:
        bandwidth = librosa.feature.spectral_bandwidth(y=samples, sr=sample_rate)[0]
        results["spectral_bandwidth_mean"] = float(np.mean(bandwidth))
        results["spectral_bandwidth_std"] = float(np.std(bandwidth))

    return results
