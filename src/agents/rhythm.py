"""Rhythm analysis agent - beat and tempo analysis."""

import numpy as np

from .base import BaseAgent, AnalysisResult


class RhythmAgent(BaseAgent):
    """Agent for rhythmic analysis."""

    name = "rhythm"
    description = "Rhythmic analysis (BPM, beats, onsets)"

    def analyse(self, samples: np.ndarray, sample_rate: int) -> AnalysisResult:
        """
        Perform rhythmic analysis.

        Extracts:
        - Tempo (BPM)
        - Beat positions
        - Onset times
        - Tempo stability
        """
        try:
            import librosa

            data = {}

            # Tempo estimation
            tempo, beat_frames = librosa.beat.beat_track(y=samples, sr=sample_rate)

            # Handle tempo being an array in newer librosa versions
            if isinstance(tempo, np.ndarray):
                tempo = float(tempo[0]) if len(tempo) > 0 else 0.0
            else:
                tempo = float(tempo)

            data["tempo_bpm"] = tempo

            # Beat times
            beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate)
            data["beat_count"] = len(beat_times)
            data["beat_times"] = [float(t) for t in beat_times[:20]]  # First 20 beats

            # Inter-beat intervals for tempo stability
            if len(beat_times) > 1:
                ibis = np.diff(beat_times)
                data["tempo_stability"] = {
                    "mean_ibi": float(np.mean(ibis)),
                    "std_ibi": float(np.std(ibis)),
                    "cv": float(np.std(ibis) / np.mean(ibis)) if np.mean(ibis) > 0 else 0,
                }
            else:
                data["tempo_stability"] = {"mean_ibi": 0, "std_ibi": 0, "cv": 0}

            # Onset detection
            onset_frames = librosa.onset.onset_detect(y=samples, sr=sample_rate)
            onset_times = librosa.frames_to_time(onset_frames, sr=sample_rate)
            data["onset_count"] = len(onset_times)
            data["onset_times"] = [float(t) for t in onset_times[:30]]  # First 30 onsets

            # Onset strength
            onset_env = librosa.onset.onset_strength(y=samples, sr=sample_rate)
            data["onset_strength"] = {
                "mean": float(np.mean(onset_env)),
                "max": float(np.max(onset_env)),
                "std": float(np.std(onset_env)),
            }

            # Tempogram for tempo variations
            tempogram = librosa.feature.tempogram(y=samples, sr=sample_rate)
            data["tempo_range"] = {
                "dominant_tempos": self._get_dominant_tempos(tempogram, sample_rate),
            }

            return AnalysisResult(agent=self.name, success=True, data=data)

        except Exception as e:
            return AnalysisResult(
                agent=self.name, success=False, error=str(e)
            )

    def _get_dominant_tempos(self, tempogram: np.ndarray, sr: int) -> list[float]:
        """Extract dominant tempos from tempogram."""
        import librosa

        # Average tempogram over time
        avg_tempogram = np.mean(tempogram, axis=1)

        # Get BPM axis
        bpms = librosa.tempo_frequencies(tempogram.shape[0], sr=sr)

        # Find peaks (filter out inf and zero values)
        top_indices = np.argsort(avg_tempogram)[-3:][::-1]
        return [float(bpms[i]) for i in top_indices if np.isfinite(bpms[i]) and bpms[i] > 0]
