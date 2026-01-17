"""Temporal analysis agent - time domain analysis."""

import numpy as np

from .base import BaseAgent, AnalysisResult
from .registry import register_agent


@register_agent
class TemporalAgent(BaseAgent):
    """Agent for time domain analysis."""

    name = "temporal"
    description = "Time domain analysis (waveform, envelope, dynamics)"

    def analyse(self, samples: np.ndarray, sample_rate: int) -> AnalysisResult:
        """
        Perform temporal analysis.

        Extracts:
        - Duration and sample count
        - Amplitude statistics
        - RMS energy over time
        - Zero-crossing rate
        - Dynamic range
        """
        try:
            import librosa

            data = {}

            # Basic info
            duration = len(samples) / sample_rate
            data["duration_seconds"] = float(duration)
            data["sample_count"] = len(samples)

            # Amplitude statistics
            data["amplitude"] = {
                "min": float(np.min(samples)),
                "max": float(np.max(samples)),
                "mean": float(np.mean(samples)),
                "std": float(np.std(samples)),
                "peak_to_peak": float(np.max(samples) - np.min(samples)),
            }

            # RMS energy - perceived loudness over time
            rms = librosa.feature.rms(y=samples)[0]
            data["rms_energy"] = {
                "mean": float(np.mean(rms)),
                "std": float(np.std(rms)),
                "min": float(np.min(rms)),
                "max": float(np.max(rms)),
            }

            # Dynamic range in dB
            rms_db = librosa.amplitude_to_db(rms)
            data["dynamic_range_db"] = float(np.max(rms_db) - np.min(rms_db))

            # Zero-crossing rate - indication of noisiness/pitch
            zcr = librosa.feature.zero_crossing_rate(samples)[0]
            data["zero_crossing_rate"] = {
                "mean": float(np.mean(zcr)),
                "std": float(np.std(zcr)),
            }

            # Envelope analysis
            envelope = np.abs(librosa.effects.preemphasis(samples))
            data["envelope"] = {
                "attack_samples": int(np.argmax(envelope[:len(envelope) // 4])),
                "peak_position": float(np.argmax(envelope) / len(envelope)),
            }

            # Silence detection
            intervals = librosa.effects.split(samples, top_db=30)
            data["non_silent_intervals"] = len(intervals)
            if len(intervals) > 0:
                total_non_silent = sum(end - start for start, end in intervals)
                data["non_silent_ratio"] = float(total_non_silent / len(samples))
            else:
                data["non_silent_ratio"] = 0.0

            return AnalysisResult(agent=self.name, success=True, data=data)

        except Exception as e:
            return AnalysisResult(
                agent=self.name, success=False, error=str(e)
            )
