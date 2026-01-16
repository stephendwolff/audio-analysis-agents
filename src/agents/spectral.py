"""Spectral analysis agent - frequency domain analysis."""

import numpy as np

from .base import BaseAgent, AnalysisResult


class SpectralAgent(BaseAgent):
    """Agent for frequency domain analysis."""

    name = "spectral"
    description = "Frequency domain analysis (FFT, spectrograms, spectral features)"

    def analyse(self, samples: np.ndarray, sample_rate: int) -> AnalysisResult:
        """
        Perform spectral analysis.

        Extracts:
        - Spectral centroid (brightness)
        - Spectral bandwidth (spread)
        - Spectral rolloff (high frequency content)
        - Dominant frequencies
        - MFCC summary
        """
        try:
            import librosa

            data = {}

            # Spectral centroid - perceived brightness
            centroid = librosa.feature.spectral_centroid(y=samples, sr=sample_rate)[0]
            data["spectral_centroid"] = {
                "mean": float(np.mean(centroid)),
                "std": float(np.std(centroid)),
                "min": float(np.min(centroid)),
                "max": float(np.max(centroid)),
            }

            # Spectral bandwidth - spread around centroid
            bandwidth = librosa.feature.spectral_bandwidth(y=samples, sr=sample_rate)[0]
            data["spectral_bandwidth"] = {
                "mean": float(np.mean(bandwidth)),
                "std": float(np.std(bandwidth)),
            }

            # Spectral rolloff - frequency below which 85% of energy lies
            rolloff = librosa.feature.spectral_rolloff(y=samples, sr=sample_rate)[0]
            data["spectral_rolloff"] = {
                "mean": float(np.mean(rolloff)),
                "std": float(np.std(rolloff)),
            }

            # Spectral flatness - tonal vs noisy
            flatness = librosa.feature.spectral_flatness(y=samples)[0]
            data["spectral_flatness"] = {
                "mean": float(np.mean(flatness)),
                "interpretation": "tonal" if np.mean(flatness) < 0.1 else "noisy",
            }

            # Dominant frequencies via FFT
            fft = np.abs(np.fft.rfft(samples))
            freqs = np.fft.rfftfreq(len(samples), 1 / sample_rate)
            top_indices = np.argsort(fft)[-5:][::-1]
            data["dominant_frequencies"] = [
                {"frequency_hz": float(freqs[i]), "magnitude": float(fft[i])}
                for i in top_indices
            ]

            # MFCCs - timbral texture
            mfccs = librosa.feature.mfcc(y=samples, sr=sample_rate, n_mfcc=13)
            data["mfcc_summary"] = {
                f"mfcc_{i}": {"mean": float(np.mean(mfccs[i])), "std": float(np.std(mfccs[i]))}
                for i in range(min(5, len(mfccs)))  # First 5 MFCCs
            }

            return AnalysisResult(agent=self.name, success=True, data=data)

        except Exception as e:
            return AnalysisResult(
                agent=self.name, success=False, error=str(e)
            )
