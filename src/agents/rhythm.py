"""Rhythm analysis agent - beat and tempo analysis."""

import numpy as np

from .base import BaseAgent, AnalysisResult
from .registry import register_agent


@register_agent
class RhythmAgent(BaseAgent):
    """Agent for rhythm and tempo analysis."""

    name = "rhythm"
    description = (
        "Analyse tempo and rhythmic properties of audio. Returns estimated BPM, "
        "beat positions, onset times, tempo stability, time signature, swing, "
        "steadiness, and upbeat detection. "
        "Use for questions about tempo, BPM, beats, rhythm, or timing."
    )

    def analyse(self, samples: np.ndarray, sample_rate: int) -> AnalysisResult:
        """
        Perform rhythmic analysis.

        Extracts:
        - Tempo (BPM)
        - Beat positions
        - Onset times
        - Tempo stability
        - Time signature
        - Swing
        - Steadiness
        - Upbeat detection
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
            data["beat_times"] = [float(t) for t in beat_times]

            # Inter-beat intervals for tempo stability
            if len(beat_times) > 1:
                ibis = np.diff(beat_times)
                cv = float(np.std(ibis) / np.mean(ibis)) if np.mean(ibis) > 0 else 0.0
                data["tempo_stability"] = {
                    "mean_ibi": float(np.mean(ibis)),
                    "std_ibi": float(np.std(ibis)),
                    "cv": cv,
                }
            else:
                ibis = np.array([])
                cv = 0.0
                data["tempo_stability"] = {"mean_ibi": 0, "std_ibi": 0, "cv": 0}

            # Onset detection
            onset_frames = librosa.onset.onset_detect(y=samples, sr=sample_rate)
            onset_times = librosa.frames_to_time(onset_frames, sr=sample_rate)
            data["onset_count"] = len(onset_times)
            data["onset_times"] = [float(t) for t in onset_times]

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

            # New dimensions for Musical Sketchbook
            data["time_signature"] = self._detect_time_signature(
                beat_times, onset_times, onset_env, sample_rate
            )
            data["swing"] = self._compute_swing(beat_times, onset_times)
            data["steadiness"] = self._compute_steadiness(cv)
            data["upbeat"] = self._detect_upbeat(beat_times, onset_times)

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

    def _detect_time_signature(self, beat_times, onset_times, onset_env, sample_rate):
        """Detect time signature by grouping onset strength around beats."""
        import librosa

        if len(beat_times) < 4:
            return "4/4"

        onsets_per_beat = []
        for i in range(len(beat_times) - 1):
            start, end = beat_times[i], beat_times[i + 1]
            count = np.sum((onset_times >= start) & (onset_times < end))
            onsets_per_beat.append(count)

        if not onsets_per_beat:
            return "4/4"

        beat_frames = librosa.time_to_frames(beat_times, sr=sample_rate)
        valid_frames = beat_frames[beat_frames < len(onset_env)]

        if len(valid_frames) < 4:
            return "4/4"

        strengths = onset_env[valid_frames]

        if len(strengths) >= 6:
            mean_str = np.mean(strengths)
            accents = strengths > mean_str

            score_3 = 0
            for i in range(len(accents) - 3):
                if accents[i] == accents[i + 3]:
                    score_3 += 1

            score_4 = 0
            for i in range(len(accents) - 4):
                if accents[i] == accents[i + 4]:
                    score_4 += 1

            score_3 = score_3 / max(len(accents) - 3, 1)
            score_4 = score_4 / max(len(accents) - 4, 1)

            if score_3 > score_4 + 0.15:
                return "3/4"

        return "4/4"

    def _compute_swing(self, beat_times, onset_times):
        """Compute swing from onset timing between beats. 0.0=straight, 1.0=triplet."""
        if len(beat_times) < 3 or len(onset_times) < 2:
            return 0.0

        ratios = []
        for i in range(len(beat_times) - 1):
            start = beat_times[i]
            end = beat_times[i + 1]
            beat_dur = end - start
            if beat_dur <= 0:
                continue
            mid_onsets = onset_times[
                (onset_times > start + beat_dur * 0.1)
                & (onset_times < end - beat_dur * 0.1)
            ]
            if len(mid_onsets) == 0:
                continue
            first_mid = mid_onsets[0]
            position = (first_mid - start) / beat_dur
            if 0.3 < position < 0.85:
                swing_ratio = max(0.0, (position - 0.5) / 0.167)
                ratios.append(min(1.0, swing_ratio))

        if not ratios:
            return 0.0

        return round(float(np.median(ratios)), 3)

    def _compute_steadiness(self, cv):
        """Convert tempo stability CV to 0-1 steadiness. Low CV = high steadiness."""
        return round(max(0.0, min(1.0, 1.0 - (cv / 0.5))), 3)

    def _detect_upbeat(self, beat_times, onset_times):
        """Detect anacrusis. True if onsets before first beat."""
        if len(beat_times) < 2 or len(onset_times) == 0:
            return False

        first_beat = beat_times[0]
        beat_interval = beat_times[1] - beat_times[0]
        pre_beat_onsets = onset_times[onset_times < first_beat]

        if len(pre_beat_onsets) == 0:
            return False

        earliest = pre_beat_onsets[0]
        gap = first_beat - earliest
        return bool(gap > beat_interval * 0.25)
