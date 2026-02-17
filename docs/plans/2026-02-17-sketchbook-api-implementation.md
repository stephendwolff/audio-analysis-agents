# Musical Sketchbook API Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add three endpoints (`POST /api/analyse`, `GET /api/analyse/{fragment_id}`, `GET /api/health`) that serve the Musical Sketchbook iOS app's analysis contract v0.2.0.

**Architecture:** A new `src/sketchbook/` Django app with a `Fragment` model, Celery task, and rule-based descriptors. The existing `RhythmAgent` gains four new fields (time_signature, swing, steadiness, upbeat). Audio decoding extends to handle mp4/webm via pydub/ffmpeg.

**Tech Stack:** Python 3.10+, Django 5, DRF, Celery, librosa, pydub, pytest

---

### Task 1: Extend load_audio() with pydub conversion

**Files:**
- Modify: `src/tools/loader.py`
- Test: `tests/test_tools/test_loader.py`

**Step 1: Write the failing test**

```python
# tests/test_tools/__init__.py
# (empty file)
```

```python
# tests/test_tools/test_loader.py
import io
import struct
import pytest
from unittest.mock import patch, MagicMock
from src.tools.loader import load_audio, convert_to_wav


def _make_wav_bytes(num_samples=100, sample_rate=22050):
    """Create minimal valid WAV bytes in memory."""
    import numpy as np
    samples = np.zeros(num_samples, dtype=np.float32)
    buf = io.BytesIO()
    import soundfile as sf
    sf.write(buf, samples, sample_rate, format="WAV")
    buf.seek(0)
    return buf.read()


class TestConvertToWav:
    """Tests for the convert_to_wav helper."""

    def test_converts_mp4_to_wav_bytes(self):
        """convert_to_wav returns a BytesIO of WAV data."""
        fake_input = io.BytesIO(b"fake-mp4-data")
        wav_bytes = _make_wav_bytes()

        mock_segment = MagicMock()
        mock_segment.export = MagicMock(side_effect=lambda buf, format: buf.write(wav_bytes))

        with patch("src.tools.loader.AudioSegment") as mock_cls:
            mock_cls.from_file.return_value = mock_segment
            result = convert_to_wav(fake_input, "audio/mp4")

        assert isinstance(result, io.BytesIO)
        mock_cls.from_file.assert_called_once()

    def test_raises_on_unsupported_mime(self):
        """convert_to_wav raises ValueError for unknown mime types."""
        with pytest.raises(ValueError, match="Unsupported"):
            convert_to_wav(io.BytesIO(b"data"), "audio/xyz")


class TestLoadAudioWithMimeType:
    """Tests for load_audio with mime_type parameter."""

    def test_load_audio_with_mime_type_converts_first(self):
        """When mime_type needs conversion, convert_to_wav is called."""
        wav_bytes = _make_wav_bytes()

        with patch("src.tools.loader.convert_to_wav") as mock_convert:
            mock_convert.return_value = io.BytesIO(wav_bytes)
            result = load_audio(io.BytesIO(b"fake"), mime_type="audio/mp4")

        mock_convert.assert_called_once()
        assert result.sample_rate > 0

    def test_load_audio_without_mime_type_skips_conversion(self):
        """When no mime_type given, no conversion happens."""
        wav_bytes = _make_wav_bytes()

        with patch("src.tools.loader.convert_to_wav") as mock_convert:
            result = load_audio(io.BytesIO(wav_bytes))

        mock_convert.assert_not_called()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_tools/test_loader.py -v`
Expected: FAIL — `convert_to_wav` does not exist, `load_audio` does not accept `mime_type`

**Step 3: Write minimal implementation**

Update `src/tools/loader.py`:

```python
"""Audio file loading utilities."""

import io
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf


MIME_TO_FORMAT = {
    "audio/mp4": "mp4",
    "audio/m4a": "m4a",
    "audio/x-m4a": "m4a",
    "audio/webm": "webm",
    "audio/aac": "aac",
}

SUPPORTED_MIME_TYPES = {
    "audio/wav", "audio/wave", "audio/x-wav",
    "audio/mp3", "audio/mpeg",
    "audio/flac",
    "audio/ogg",
    *MIME_TO_FORMAT.keys(),
}


@dataclass
class AudioData:
    """Container for loaded audio data."""

    samples: np.ndarray
    sample_rate: int
    duration: float
    channels: int
    file_path: Optional[Path] = None

    @property
    def is_stereo(self) -> bool:
        return self.channels == 2

    @property
    def is_mono(self) -> bool:
        return self.channels == 1


def _is_url(path: str) -> bool:
    """Check if path is a URL."""
    return path.startswith("http://") or path.startswith("https://")


def _download_to_buffer(url: str) -> io.BytesIO:
    """Download a URL to an in-memory buffer."""
    with urllib.request.urlopen(url) as response:
        data = response.read()
    return io.BytesIO(data)


def convert_to_wav(file_obj: io.BytesIO, mime_type: str) -> io.BytesIO:
    """
    Convert audio from a container format to WAV using pydub (ffmpeg).

    Args:
        file_obj: Audio data as a BytesIO object
        mime_type: MIME type of the input audio

    Returns:
        BytesIO containing WAV data

    Raises:
        ValueError: If the mime_type is not in MIME_TO_FORMAT
    """
    fmt = MIME_TO_FORMAT.get(mime_type)
    if fmt is None:
        raise ValueError(f"Unsupported mime type for conversion: {mime_type}")

    from pydub import AudioSegment

    segment = AudioSegment.from_file(file_obj, format=fmt)
    wav_buffer = io.BytesIO()
    segment.export(wav_buffer, format="wav")
    wav_buffer.seek(0)
    return wav_buffer


def load_audio(
    file_path: "str | Path | io.BytesIO",
    target_sr: Optional[int] = None,
    mono: bool = True,
    mime_type: Optional[str] = None,
) -> AudioData:
    """
    Load an audio file from a local path, URL, or BytesIO buffer.

    Args:
        file_path: Path to audio file, URL, or BytesIO with audio data
        target_sr: Target sample rate (None to keep original)
        mono: Convert to mono if True
        mime_type: MIME type hint. If the type needs conversion (mp4, webm),
                   the audio is converted to WAV via pydub before loading.

    Returns:
        AudioData containing the loaded audio
    """
    source_path = None

    # If we have a BytesIO or file-like object
    if isinstance(file_path, io.BytesIO):
        buffer = file_path
        if mime_type and mime_type in MIME_TO_FORMAT:
            buffer = convert_to_wav(buffer, mime_type)
        samples, sample_rate = sf.read(buffer, always_2d=True)
    elif _is_url(str(file_path)):
        try:
            buffer = _download_to_buffer(str(file_path))
            if mime_type and mime_type in MIME_TO_FORMAT:
                buffer = convert_to_wav(buffer, mime_type)
            samples, sample_rate = sf.read(buffer, always_2d=True)
        except Exception as e:
            raise FileNotFoundError(f"Could not load audio from URL: {file_path}. Error: {e}")
    else:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        if mime_type and mime_type in MIME_TO_FORMAT:
            with open(file_path, "rb") as f:
                buffer = convert_to_wav(io.BytesIO(f.read()), mime_type)
            samples, sample_rate = sf.read(buffer, always_2d=True)
        else:
            samples, sample_rate = sf.read(file_path, always_2d=True)
            source_path = file_path

    # Convert to mono if requested
    if mono and samples.shape[1] > 1:
        samples = np.mean(samples, axis=1)
        channels = 1
    else:
        channels = samples.shape[1] if samples.ndim > 1 else 1
        if samples.ndim > 1 and samples.shape[1] == 1:
            samples = samples.flatten()

    # Resample if needed
    if target_sr is not None and target_sr != sample_rate:
        import librosa

        samples = librosa.resample(
            samples, orig_sr=sample_rate, target_sr=target_sr
        )
        sample_rate = target_sr

    duration = len(samples) / sample_rate

    return AudioData(
        samples=samples,
        sample_rate=sample_rate,
        duration=duration,
        channels=channels,
        file_path=source_path,
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_tools/test_loader.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/tools/loader.py tests/test_tools/
git commit -m "feat(tools): extend load_audio with mime_type conversion via pydub"
```

---

### Task 2: Add pydub dependency

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add pydub to the api optional dependency group**

In `pyproject.toml`, add `"pydub>=0.25.0"` to the `[project.optional-dependencies] api` list, after the `django-celery-results` line:

```toml
api = [
    # ... existing deps ...
    "django-celery-results>=2.5.0",
    "pydub>=0.25.0",
]
```

**Step 2: Install**

Run: `pip install pydub`

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add pydub dependency for audio format conversion"
```

---

### Task 3: Enhance RhythmAgent with new dimensions

**Files:**
- Modify: `src/agents/rhythm.py`
- Test: `tests/test_agents/test_rhythm_dimensions.py`

**Step 1: Write the failing test**

```python
# tests/test_agents/test_rhythm_dimensions.py
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
    duration = 4.0  # 4 seconds = 8 beats at 120 BPM
    samples = np.zeros(int(sr * duration), dtype=np.float32)
    beat_interval = 0.5  # 120 BPM = 0.5s per beat
    # Place short clicks at each beat
    for i in range(int(duration / beat_interval)):
        pos = int(i * beat_interval * sr)
        click_len = min(200, len(samples) - pos)
        samples[pos : pos + click_len] = 0.8 * np.sin(
            2 * np.pi * 1000 * np.arange(click_len) / sr
        )
    return samples, sr


class TestNewDimensions:
    """Test that analyse() returns the new dimension fields."""

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
        beat_interval = 0.5  # 120 BPM
        for i in range(int(duration / beat_interval)):
            pos = int(i * beat_interval * sr)
            click_len = min(200, len(samples) - pos)
            samples[pos : pos + click_len] = 0.8 * np.sin(
                2 * np.pi * 1000 * np.arange(click_len) / sr
            )
        result = agent.analyse(samples, sr)
        assert result.success
        # 30s at 120 BPM ≈ 60 beats; should not be truncated to 20
        assert len(result.data["beat_times"]) > 20

    def test_onset_times_not_truncated(self, agent):
        """onset_times should include all onsets, not just first 30."""
        sr = 22050
        duration = 30.0
        samples = np.zeros(int(sr * duration), dtype=np.float32)
        beat_interval = 0.25  # Fast onsets
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
        """A perfectly regular click track should have high steadiness."""
        samples, sr = click_track_120bpm
        result = agent.analyse(samples, sr)
        assert result.success
        assert result.data["steadiness"] >= 0.5

    def test_straight_click_has_low_swing(self, agent, click_track_120bpm):
        """Evenly spaced clicks should have low swing value."""
        samples, sr = click_track_120bpm
        result = agent.analyse(samples, sr)
        assert result.success
        assert result.data["swing"] <= 0.5
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_agents/test_rhythm_dimensions.py -v`
Expected: FAIL — new keys missing from result.data

**Step 3: Write minimal implementation**

Update `src/agents/rhythm.py` — add four helper methods and call them from `analyse()`:

```python
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
        """Perform rhythmic analysis."""
        try:
            import librosa

            data = {}

            # Tempo estimation
            tempo, beat_frames = librosa.beat.beat_track(y=samples, sr=sample_rate)
            if isinstance(tempo, np.ndarray):
                tempo = float(tempo[0]) if len(tempo) > 0 else 0.0
            else:
                tempo = float(tempo)
            data["tempo_bpm"] = tempo

            # Beat times (full array — no truncation)
            beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate)
            data["beat_count"] = len(beat_times)
            data["beat_times"] = [float(t) for t in beat_times]

            # Inter-beat intervals for tempo stability
            if len(beat_times) > 1:
                ibis = np.diff(beat_times)
                cv = float(np.std(ibis) / np.mean(ibis)) if np.mean(ibis) > 0 else 0
                data["tempo_stability"] = {
                    "mean_ibi": float(np.mean(ibis)),
                    "std_ibi": float(np.std(ibis)),
                    "cv": cv,
                }
            else:
                ibis = np.array([])
                cv = 0.0
                data["tempo_stability"] = {"mean_ibi": 0, "std_ibi": 0, "cv": 0}

            # Onset detection (full array — no truncation)
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
            return AnalysisResult(agent=self.name, success=False, error=str(e))

    def _get_dominant_tempos(self, tempogram: np.ndarray, sr: int) -> list[float]:
        """Extract dominant tempos from tempogram."""
        import librosa

        avg_tempogram = np.mean(tempogram, axis=1)
        bpms = librosa.tempo_frequencies(tempogram.shape[0], sr=sr)
        top_indices = np.argsort(avg_tempogram)[-3:][::-1]
        return [float(bpms[i]) for i in top_indices if np.isfinite(bpms[i]) and bpms[i] > 0]

    def _detect_time_signature(
        self,
        beat_times: np.ndarray,
        onset_times: np.ndarray,
        onset_env: np.ndarray,
        sample_rate: int,
    ) -> str:
        """
        Detect time signature by grouping onset strength around beats.

        Counts strong onsets per beat cycle. Groups of 3 suggest 3/4;
        groups of 4 suggest 4/4.
        """
        import librosa

        if len(beat_times) < 4:
            return "4/4"

        # Count onsets between consecutive beats
        onsets_per_beat = []
        for i in range(len(beat_times) - 1):
            start, end = beat_times[i], beat_times[i + 1]
            count = np.sum((onset_times >= start) & (onset_times < end))
            onsets_per_beat.append(count)

        if not onsets_per_beat:
            return "4/4"

        # Look at onset strength at beat positions to find accent pattern
        # Convert beat times to frames
        beat_frames = librosa.time_to_frames(beat_times, sr=sample_rate)
        valid_frames = beat_frames[beat_frames < len(onset_env)]
        if len(valid_frames) < 4:
            return "4/4"

        strengths = onset_env[valid_frames]

        # Look for repeating accent pattern
        # Strong beats tend to have higher onset strength
        if len(strengths) >= 6:
            # Check correlation with period-3 vs period-4 pattern
            mean_str = np.mean(strengths)
            accents = strengths > mean_str

            # Score period-3: accents should repeat every 3 beats
            score_3 = 0
            for i in range(len(accents) - 3):
                if accents[i] == accents[i + 3]:
                    score_3 += 1

            # Score period-4: accents should repeat every 4 beats
            score_4 = 0
            for i in range(len(accents) - 4):
                if accents[i] == accents[i + 4]:
                    score_4 += 1

            # Normalise scores
            score_3 = score_3 / max(len(accents) - 3, 1)
            score_4 = score_4 / max(len(accents) - 4, 1)

            if score_3 > score_4 + 0.15:
                return "3/4"

        return "4/4"

    def _compute_swing(
        self, beat_times: np.ndarray, onset_times: np.ndarray
    ) -> float:
        """
        Compute swing amount from onset timing between beats.

        Compares the position of onsets between consecutive beats to
        an even subdivision. Straight = 0.0, full triplet swing = 1.0.
        """
        if len(beat_times) < 3 or len(onset_times) < 2:
            return 0.0

        ratios = []
        for i in range(len(beat_times) - 1):
            start = beat_times[i]
            end = beat_times[i + 1]
            beat_dur = end - start
            if beat_dur <= 0:
                continue

            # Find onsets between this beat and next
            mid_onsets = onset_times[
                (onset_times > start + beat_dur * 0.1)
                & (onset_times < end - beat_dur * 0.1)
            ]

            if len(mid_onsets) == 0:
                continue

            # Position of first mid-onset relative to the beat interval
            first_mid = mid_onsets[0]
            position = (first_mid - start) / beat_dur  # 0.5 = even, >0.5 = swung

            if 0.3 < position < 0.85:
                # Convert position to swing ratio
                # 0.5 → 0.0 swing, 0.667 → 1.0 swing (triplet)
                swing_ratio = max(0.0, (position - 0.5) / 0.167)
                ratios.append(min(1.0, swing_ratio))

        if not ratios:
            return 0.0

        return round(float(np.median(ratios)), 3)

    def _compute_steadiness(self, cv: float) -> float:
        """
        Convert tempo stability CV to a 0-1 steadiness score.

        Low CV (consistent tempo) → high steadiness.
        """
        return round(max(0.0, min(1.0, 1.0 - (cv / 0.5))), 3)

    def _detect_upbeat(
        self, beat_times: np.ndarray, onset_times: np.ndarray
    ) -> bool:
        """
        Detect anacrusis (pickup/upbeat).

        Returns True if significant onsets occur before the first beat.
        """
        if len(beat_times) < 2 or len(onset_times) == 0:
            return False

        first_beat = beat_times[0]
        beat_interval = beat_times[1] - beat_times[0] if len(beat_times) > 1 else 1.0

        # Check for onsets before the first beat
        pre_beat_onsets = onset_times[onset_times < first_beat]
        if len(pre_beat_onsets) == 0:
            return False

        # If the earliest onset is more than half a beat before the first beat,
        # it's likely an upbeat
        earliest = pre_beat_onsets[0]
        gap = first_beat - earliest
        return gap > beat_interval * 0.25
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_agents/test_rhythm_dimensions.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/agents/rhythm.py tests/test_agents/test_rhythm_dimensions.py
git commit -m "feat(agents): add time_signature, swing, steadiness, upbeat to RhythmAgent"
```

---

### Task 4: Create sketchbook Django app scaffold

**Files:**
- Create: `src/sketchbook/__init__.py`
- Create: `src/sketchbook/apps.py`
- Modify: `config/settings.py`

**Step 1: Create the app module**

```python
# src/sketchbook/__init__.py
```

```python
# src/sketchbook/apps.py
from django.apps import AppConfig


class SketchbookConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "src.sketchbook"
    verbose_name = "Musical Sketchbook"
```

**Step 2: Register in settings**

In `config/settings.py`, add `"src.sketchbook"` to `INSTALLED_APPS` after `"src.api"` (line 41):

```python
INSTALLED_APPS = [
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.staticfiles",
    "rest_framework",
    "rest_framework_simplejwt",
    "storages",
    "channels",
    "django_celery_results",
    "src.api",
    "src.sketchbook",
]
```

Also add the model version setting at the end of `config/settings.py`:

```python
# Musical Sketchbook
SKETCHBOOK_MODEL_VERSION = "1.0.0"
```

**Step 3: Commit**

```bash
git add src/sketchbook/ config/settings.py
git commit -m "feat(sketchbook): create Django app scaffold"
```

---

### Task 5: Create Fragment model

**Files:**
- Create: `src/sketchbook/models.py`
- Test: `tests/test_sketchbook/test_models.py`

**Step 1: Write the failing test**

```python
# tests/test_sketchbook/__init__.py
# (empty file)
```

```python
# tests/test_sketchbook/test_models.py
import uuid
import pytest
from django.test import TestCase
from django.contrib.auth.models import User
from src.sketchbook.models import Fragment


class TestFragmentModel(TestCase):
    """Tests for the Fragment model."""

    def setUp(self):
        self.user = User.objects.create_user(username="test", password="test")

    def test_create_fragment(self):
        """Can create a Fragment with required fields."""
        fid = uuid.uuid4()
        fragment = Fragment.objects.create(
            fragment_id=fid,
            user=self.user,
            audio_storage_path="test.wav",
            mime_type="audio/wav",
            duration_seconds=5.0,
            model_version="1.0.0",
        )
        assert fragment.fragment_id == fid
        assert fragment.status == Fragment.Status.PENDING

    def test_fragment_id_is_unique(self):
        """Duplicate fragment_id raises IntegrityError."""
        from django.db import IntegrityError

        fid = uuid.uuid4()
        Fragment.objects.create(
            fragment_id=fid,
            user=self.user,
            audio_storage_path="a.wav",
            mime_type="audio/wav",
            duration_seconds=5.0,
            model_version="1.0.0",
        )
        with pytest.raises(IntegrityError):
            Fragment.objects.create(
                fragment_id=fid,
                user=self.user,
                audio_storage_path="b.wav",
                mime_type="audio/wav",
                duration_seconds=5.0,
                model_version="1.0.0",
            )

    def test_default_status_is_pending(self):
        """New fragments start with status=pending."""
        fragment = Fragment.objects.create(
            fragment_id=uuid.uuid4(),
            user=self.user,
            audio_storage_path="test.wav",
            mime_type="audio/wav",
            duration_seconds=5.0,
            model_version="1.0.0",
        )
        assert fragment.status == "pending"

    def test_status_choices(self):
        """Status has the four expected values."""
        assert Fragment.Status.PENDING == "pending"
        assert Fragment.Status.ANALYZING == "analyzing"
        assert Fragment.Status.COMPLETE == "complete"
        assert Fragment.Status.FAILED == "failed"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_sketchbook/test_models.py -v`
Expected: FAIL — Fragment model does not exist

**Step 3: Write the model**

```python
# src/sketchbook/models.py
"""Models for the Musical Sketchbook API."""

from django.contrib.auth.models import User
from django.db import models


class Fragment(models.Model):
    """An audio fragment submitted by the Musical Sketchbook iOS app."""

    fragment_id = models.UUIDField(unique=True, db_index=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="fragments")

    audio_storage_path = models.CharField(max_length=500)
    mime_type = models.CharField(max_length=100)
    duration_seconds = models.FloatField()

    class Status(models.TextChoices):
        PENDING = "pending"
        ANALYZING = "analyzing"
        COMPLETE = "complete"
        FAILED = "failed"

    status = models.CharField(max_length=20, choices=Status, default=Status.PENDING)
    error_message = models.TextField(blank=True)

    model_version = models.CharField(max_length=20)
    analysis = models.JSONField(default=dict)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Fragment {self.fragment_id} ({self.status})"
```

**Step 4: Update test conftest to include sketchbook app**

Add `"src.sketchbook"` to `INSTALLED_APPS` in `tests/conftest.py`:

```python
INSTALLED_APPS=[
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "src.api",
    "src.sketchbook",
],
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_sketchbook/test_models.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/sketchbook/models.py tests/test_sketchbook/ tests/conftest.py
git commit -m "feat(sketchbook): add Fragment model"
```

---

### Task 6: Create descriptor generator

**Files:**
- Create: `src/sketchbook/descriptors.py`
- Test: `tests/test_sketchbook/test_descriptors.py`

**Step 1: Write the failing test**

```python
# tests/test_sketchbook/test_descriptors.py
"""Tests for rule-based descriptor generation."""

import pytest
from src.sketchbook.descriptors import generate_descriptors


class TestGenerateDescriptors:

    def test_fast_tempo_returns_driving(self):
        dims = {"bpm": 150, "swing": 0.0, "steadiness": 0.5, "upbeat": False}
        assert "driving" in generate_descriptors(dims)

    def test_slow_tempo_returns_laid_back(self):
        dims = {"bpm": 80, "swing": 0.0, "steadiness": 0.5, "upbeat": False}
        assert "laid-back" in generate_descriptors(dims)

    def test_moderate_tempo(self):
        dims = {"bpm": 110, "swing": 0.0, "steadiness": 0.5, "upbeat": False}
        assert "moderate-tempo" in generate_descriptors(dims)

    def test_high_swing_returns_swung(self):
        dims = {"bpm": 110, "swing": 0.6, "steadiness": 0.5, "upbeat": False}
        assert "swung" in generate_descriptors(dims)

    def test_low_swing_returns_straight(self):
        dims = {"bpm": 110, "swing": 0.05, "steadiness": 0.5, "upbeat": False}
        assert "straight" in generate_descriptors(dims)

    def test_high_steadiness_returns_steady(self):
        dims = {"bpm": 110, "swing": 0.0, "steadiness": 0.9, "upbeat": False}
        assert "steady" in generate_descriptors(dims)

    def test_low_steadiness_returns_loose(self):
        dims = {"bpm": 110, "swing": 0.0, "steadiness": 0.3, "upbeat": False}
        assert "loose" in generate_descriptors(dims)

    def test_upbeat_returns_upbeat_start(self):
        dims = {"bpm": 110, "swing": 0.0, "steadiness": 0.5, "upbeat": True}
        assert "upbeat-start" in generate_descriptors(dims)

    def test_multiple_descriptors_combine(self):
        dims = {"bpm": 80, "swing": 0.6, "steadiness": 0.3, "upbeat": False}
        result = generate_descriptors(dims)
        assert "laid-back" in result
        assert "swung" in result
        assert "loose" in result

    def test_returns_list_of_strings(self):
        dims = {"bpm": 110, "swing": 0.2, "steadiness": 0.5, "upbeat": False}
        result = generate_descriptors(dims)
        assert isinstance(result, list)
        assert all(isinstance(d, str) for d in result)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_sketchbook/test_descriptors.py -v`
Expected: FAIL — module does not exist

**Step 3: Write minimal implementation**

```python
# src/sketchbook/descriptors.py
"""Rule-based descriptor generation from analysis dimensions."""


def generate_descriptors(dimensions: dict) -> list[str]:
    """
    Generate human-readable rhythm descriptors from dimension values.

    Args:
        dimensions: Dict with keys bpm, swing, steadiness, upbeat

    Returns:
        List of descriptor strings
    """
    descriptors = []
    bpm = dimensions.get("bpm", 0)
    swing = dimensions.get("swing", 0)
    steadiness = dimensions.get("steadiness", 0)
    upbeat = dimensions.get("upbeat", False)

    # Tempo
    if bpm > 140:
        descriptors.append("driving")
    elif bpm < 90:
        descriptors.append("laid-back")
    else:
        descriptors.append("moderate-tempo")

    # Swing
    if swing > 0.4:
        descriptors.append("swung")
    elif swing < 0.1:
        descriptors.append("straight")

    # Steadiness
    if steadiness > 0.8:
        descriptors.append("steady")
    elif steadiness < 0.4:
        descriptors.append("loose")

    # Upbeat
    if upbeat:
        descriptors.append("upbeat-start")

    return descriptors
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_sketchbook/test_descriptors.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/sketchbook/descriptors.py tests/test_sketchbook/test_descriptors.py
git commit -m "feat(sketchbook): add rule-based descriptor generation"
```

---

### Task 7: Create response builder (serializers.py)

**Files:**
- Create: `src/sketchbook/serializers.py`
- Test: `tests/test_sketchbook/test_serializers.py`

**Step 1: Write the failing test**

```python
# tests/test_sketchbook/test_serializers.py
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_sketchbook/test_serializers.py -v`
Expected: FAIL — module does not exist

**Step 3: Write minimal implementation**

```python
# src/sketchbook/serializers.py
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_sketchbook/test_serializers.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/sketchbook/serializers.py tests/test_sketchbook/test_serializers.py
git commit -m "feat(sketchbook): add response builder for contract shape"
```

---

### Task 8: Create analyse_fragment Celery task

**Files:**
- Create: `src/sketchbook/tasks.py`
- Test: `tests/test_sketchbook/test_tasks.py`

**Step 1: Write the failing test**

```python
# tests/test_sketchbook/test_tasks.py
"""Tests for the analyse_fragment Celery task."""

import uuid
import pytest
from unittest.mock import patch, MagicMock
from django.test import TestCase
from django.contrib.auth.models import User
from src.sketchbook.models import Fragment
from src.sketchbook.tasks import analyse_fragment
from src.agents.base import AnalysisResult


class TestAnalyseFragmentTask(TestCase):

    def setUp(self):
        self.user = User.objects.create_user(username="test", password="test")
        self.fragment = Fragment.objects.create(
            fragment_id=uuid.uuid4(),
            user=self.user,
            audio_storage_path="/tmp/test.wav",
            mime_type="audio/wav",
            duration_seconds=5.0,
            model_version="1.0.0",
        )

    @patch("src.sketchbook.tasks.TemporalAgent")
    @patch("src.sketchbook.tasks.SpectralAgent")
    @patch("src.sketchbook.tasks.RhythmAgent")
    @patch("src.sketchbook.tasks.load_audio")
    def test_sets_status_to_complete_on_success(
        self, mock_load, mock_rhythm_cls, mock_spectral_cls, mock_temporal_cls
    ):
        mock_audio = MagicMock()
        mock_audio.samples = MagicMock()
        mock_audio.sample_rate = 22050
        mock_load.return_value = mock_audio

        mock_rhythm_cls.return_value.analyse.return_value = AnalysisResult(
            agent="rhythm", success=True, data={
                "tempo_bpm": 120, "time_signature": "4/4", "swing": 0.1,
                "steadiness": 0.9, "upbeat": False,
                "beat_times": [0.5, 1.0], "onset_times": [0.1, 0.5],
            }
        )
        mock_spectral_cls.return_value.analyse.return_value = AnalysisResult(
            agent="spectral", success=True,
            data={"spectral_centroid": {"mean": 2000}},
        )
        mock_temporal_cls.return_value.analyse.return_value = AnalysisResult(
            agent="temporal", success=True,
            data={"rms_energy": {"mean": 0.05}},
        )

        analyse_fragment(str(self.fragment.fragment_id))

        self.fragment.refresh_from_db()
        assert self.fragment.status == Fragment.Status.COMPLETE
        assert "dimensions" in self.fragment.analysis
        assert self.fragment.analysis["dimensions"]["bpm"] == 120

    @patch("src.sketchbook.tasks.load_audio")
    def test_sets_status_to_failed_on_error(self, mock_load):
        mock_load.side_effect = FileNotFoundError("audio not found")

        analyse_fragment(str(self.fragment.fragment_id))

        self.fragment.refresh_from_db()
        assert self.fragment.status == Fragment.Status.FAILED
        assert "audio not found" in self.fragment.error_message

    @patch("src.sketchbook.tasks.TemporalAgent")
    @patch("src.sketchbook.tasks.SpectralAgent")
    @patch("src.sketchbook.tasks.RhythmAgent")
    @patch("src.sketchbook.tasks.load_audio")
    def test_sets_status_to_analyzing_during_run(
        self, mock_load, mock_rhythm_cls, mock_spectral_cls, mock_temporal_cls
    ):
        statuses = []

        def capture_status(*args, **kwargs):
            self.fragment.refresh_from_db()
            statuses.append(self.fragment.status)
            return AnalysisResult(
                agent="rhythm", success=True, data={
                    "tempo_bpm": 120, "time_signature": "4/4", "swing": 0.0,
                    "steadiness": 0.9, "upbeat": False,
                    "beat_times": [], "onset_times": [],
                }
            )

        mock_audio = MagicMock()
        mock_load.return_value = mock_audio
        mock_rhythm_cls.return_value.analyse.side_effect = capture_status
        mock_spectral_cls.return_value.analyse.return_value = AnalysisResult(
            agent="spectral", success=True, data={"spectral_centroid": {"mean": 0}},
        )
        mock_temporal_cls.return_value.analyse.return_value = AnalysisResult(
            agent="temporal", success=True, data={"rms_energy": {"mean": 0}},
        )

        analyse_fragment(str(self.fragment.fragment_id))
        assert "analyzing" in statuses
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_sketchbook/test_tasks.py -v`
Expected: FAIL — module does not exist

**Step 3: Write minimal implementation**

```python
# src/sketchbook/tasks.py
"""Celery tasks for Musical Sketchbook fragment analysis."""

import logging

from celery import shared_task

from src.agents.rhythm import RhythmAgent
from src.agents.spectral import SpectralAgent
from src.agents.temporal import TemporalAgent
from src.tools.loader import load_audio
from .models import Fragment
from .serializers import build_analysis_result

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=3, default_retry_delay=30)
def analyse_fragment(self, fragment_id_str: str):
    """
    Run analysis agents on a fragment and store the result.

    Args:
        fragment_id_str: The fragment_id UUID as a string
    """
    try:
        fragment = Fragment.objects.get(fragment_id=fragment_id_str)
    except Fragment.DoesNotExist:
        logger.error(f"Fragment {fragment_id_str} not found")
        return

    fragment.status = Fragment.Status.ANALYZING
    fragment.save(update_fields=["status"])

    try:
        audio = load_audio(
            fragment.audio_storage_path,
            target_sr=22050,
            mono=True,
            mime_type=fragment.mime_type,
        )

        rhythm = RhythmAgent().analyse(audio.samples, audio.sample_rate)
        spectral = SpectralAgent().analyse(audio.samples, audio.sample_rate)
        temporal = TemporalAgent().analyse(audio.samples, audio.sample_rate)

        if not rhythm.success:
            raise RuntimeError(f"Rhythm analysis failed: {rhythm.error}")
        if not spectral.success:
            raise RuntimeError(f"Spectral analysis failed: {spectral.error}")
        if not temporal.success:
            raise RuntimeError(f"Temporal analysis failed: {temporal.error}")

        fragment.analysis = build_analysis_result(rhythm, spectral, temporal)
        fragment.status = Fragment.Status.COMPLETE
        fragment.save(update_fields=["analysis", "status"])

        logger.info(f"Fragment {fragment_id_str} analysis complete")

    except Exception as e:
        logger.exception(f"Error analysing fragment {fragment_id_str}: {e}")
        fragment.status = Fragment.Status.FAILED
        fragment.error_message = str(e)
        fragment.save(update_fields=["status", "error_message"])
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_sketchbook/test_tasks.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/sketchbook/tasks.py tests/test_sketchbook/test_tasks.py
git commit -m "feat(sketchbook): add analyse_fragment Celery task"
```

---

### Task 9: Create API views and URL routing

**Files:**
- Create: `src/sketchbook/views.py`
- Create: `src/sketchbook/urls.py`
- Modify: `config/urls.py`
- Test: `tests/test_sketchbook/test_views.py`

**Step 1: Write the failing test**

```python
# tests/test_sketchbook/test_views.py
"""Tests for the sketchbook API views."""

import uuid
import pytest
from unittest.mock import patch, MagicMock
from django.test import TestCase
from django.contrib.auth.models import User
from rest_framework.test import APIClient
from src.sketchbook.models import Fragment


class TestHealthView(TestCase):

    def test_health_returns_ok(self):
        client = APIClient()
        response = client.get("/api/health/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "agents" in data
        assert "rhythm" in data["agents"]


class TestAnalysePostView(TestCase):

    def setUp(self):
        self.user = User.objects.create_user(username="test", password="test")
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)

    @patch("src.sketchbook.views.analyse_fragment")
    def test_post_returns_202(self, mock_task):
        mock_task.delay.return_value = MagicMock(id="task-123")
        from django.core.files.uploadedfile import SimpleUploadedFile

        audio = SimpleUploadedFile("test.wav", b"fake-audio-data", content_type="audio/wav")
        response = self.client.post(
            "/api/analyse/",
            {"audio": audio, "mime_type": "audio/wav",
             "fragment_id": str(uuid.uuid4()), "duration_seconds": "5.0"},
            format="multipart",
        )
        assert response.status_code == 202
        data = response.json()
        assert data["status"] == "pending"
        assert "poll_url" in data

    def test_post_missing_audio_returns_400(self):
        response = self.client.post(
            "/api/analyse/",
            {"mime_type": "audio/wav",
             "fragment_id": str(uuid.uuid4()), "duration_seconds": "5.0"},
            format="multipart",
        )
        assert response.status_code == 400

    def test_post_unsupported_mime_returns_422(self):
        from django.core.files.uploadedfile import SimpleUploadedFile

        audio = SimpleUploadedFile("test.xyz", b"data", content_type="audio/xyz")
        response = self.client.post(
            "/api/analyse/",
            {"audio": audio, "mime_type": "audio/xyz",
             "fragment_id": str(uuid.uuid4()), "duration_seconds": "5.0"},
            format="multipart",
        )
        assert response.status_code == 422

    @patch("src.sketchbook.views.analyse_fragment")
    def test_post_existing_complete_fragment_returns_200(self, mock_task):
        fid = uuid.uuid4()
        Fragment.objects.create(
            fragment_id=fid, user=self.user,
            audio_storage_path="test.wav", mime_type="audio/wav",
            duration_seconds=5.0, model_version="1.0.0",
            status=Fragment.Status.COMPLETE,
            analysis={"dimensions": {"bpm": 120}, "descriptors": [], "raw_data": {}},
        )
        from django.core.files.uploadedfile import SimpleUploadedFile

        audio = SimpleUploadedFile("test.wav", b"data", content_type="audio/wav")
        response = self.client.post(
            "/api/analyse/",
            {"audio": audio, "mime_type": "audio/wav",
             "fragment_id": str(fid), "duration_seconds": "5.0"},
            format="multipart",
        )
        assert response.status_code == 200
        assert response.json()["dimensions"]["bpm"] == 120

    @patch("src.sketchbook.views.analyse_fragment")
    def test_post_existing_pending_fragment_returns_202(self, mock_task):
        fid = uuid.uuid4()
        Fragment.objects.create(
            fragment_id=fid, user=self.user,
            audio_storage_path="test.wav", mime_type="audio/wav",
            duration_seconds=5.0, model_version="1.0.0",
            status=Fragment.Status.PENDING,
        )
        from django.core.files.uploadedfile import SimpleUploadedFile

        audio = SimpleUploadedFile("test.wav", b"data", content_type="audio/wav")
        response = self.client.post(
            "/api/analyse/",
            {"audio": audio, "mime_type": "audio/wav",
             "fragment_id": str(fid), "duration_seconds": "5.0"},
            format="multipart",
        )
        assert response.status_code == 202

    def test_post_requires_auth(self):
        client = APIClient()  # unauthenticated
        from django.core.files.uploadedfile import SimpleUploadedFile

        audio = SimpleUploadedFile("test.wav", b"data", content_type="audio/wav")
        response = client.post(
            "/api/analyse/",
            {"audio": audio, "mime_type": "audio/wav",
             "fragment_id": str(uuid.uuid4()), "duration_seconds": "5.0"},
            format="multipart",
        )
        assert response.status_code in (401, 403)


class TestAnalyseGetView(TestCase):

    def setUp(self):
        self.user = User.objects.create_user(username="test", password="test")
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)

    def test_get_complete_fragment(self):
        fid = uuid.uuid4()
        Fragment.objects.create(
            fragment_id=fid, user=self.user,
            audio_storage_path="test.wav", mime_type="audio/wav",
            duration_seconds=5.0, model_version="1.0.0",
            status=Fragment.Status.COMPLETE,
            analysis={"dimensions": {"bpm": 120}, "descriptors": ["steady"], "raw_data": {}},
        )
        response = self.client.get(f"/api/analyse/{fid}/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "complete"
        assert data["dimensions"]["bpm"] == 120
        assert data["model_version"] == "1.0.0"

    def test_get_pending_fragment(self):
        fid = uuid.uuid4()
        Fragment.objects.create(
            fragment_id=fid, user=self.user,
            audio_storage_path="test.wav", mime_type="audio/wav",
            duration_seconds=5.0, model_version="1.0.0",
            status=Fragment.Status.PENDING,
        )
        response = self.client.get(f"/api/analyse/{fid}/")
        assert response.status_code == 200
        assert response.json()["status"] == "pending"

    def test_get_failed_fragment(self):
        fid = uuid.uuid4()
        Fragment.objects.create(
            fragment_id=fid, user=self.user,
            audio_storage_path="test.wav", mime_type="audio/wav",
            duration_seconds=5.0, model_version="1.0.0",
            status=Fragment.Status.FAILED, error_message="Timeout",
        )
        response = self.client.get(f"/api/analyse/{fid}/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        assert data["error"] == "Timeout"

    def test_get_unknown_fragment_returns_404(self):
        response = self.client.get(f"/api/analyse/{uuid.uuid4()}/")
        assert response.status_code == 404
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_sketchbook/test_views.py -v`
Expected: FAIL — views/urls do not exist

**Step 3: Write minimal implementation**

```python
# src/sketchbook/views.py
"""API views for the Musical Sketchbook analysis endpoints."""

import logging
import uuid

from django.conf import settings
from django.core.files.storage import default_storage
from rest_framework import status
from rest_framework.parsers import MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from src.agents.registry import get_all_agents
from src.tools.loader import SUPPORTED_MIME_TYPES
from .models import Fragment
from .tasks import analyse_fragment

logger = logging.getLogger(__name__)


class HealthView(APIView):
    """GET /api/health/ — server availability check."""

    permission_classes = []
    authentication_classes = []

    def get(self, request):
        try:
            agents = [a.name for a in get_all_agents()]
        except Exception:
            return Response(
                {"status": "error", "message": "Agents unavailable"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        if not agents:
            return Response(
                {"status": "error", "message": "No agents registered"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        return Response({
            "status": "ok",
            "version": getattr(settings, "SKETCHBOOK_MODEL_VERSION", "0.0.0"),
            "agents": agents,
        })


class AnalyseView(APIView):
    """
    POST /api/analyse/ — submit a fragment for analysis.
    GET /api/analyse/{fragment_id}/ — poll for results.
    """

    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser]

    def post(self, request):
        # Validate required fields
        audio = request.FILES.get("audio")
        mime_type = request.data.get("mime_type", "")
        fragment_id_str = request.data.get("fragment_id", "")
        duration_str = request.data.get("duration_seconds", "")

        if not audio:
            return Response(
                {"error": "validation_error", "message": "audio file is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if not fragment_id_str:
            return Response(
                {"error": "validation_error", "message": "fragment_id is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if not duration_str:
            return Response(
                {"error": "validation_error", "message": "duration_seconds is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Validate fragment_id is a UUID
        try:
            fragment_id = uuid.UUID(fragment_id_str)
        except ValueError:
            return Response(
                {"error": "validation_error", "message": "fragment_id must be a valid UUID"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Validate duration
        try:
            duration_seconds = float(duration_str)
        except (ValueError, TypeError):
            return Response(
                {"error": "validation_error", "message": "duration_seconds must be a number"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Validate mime_type
        if mime_type not in SUPPORTED_MIME_TYPES:
            return Response(
                {"error": "unsupported_format", "message": f"Cannot decode {mime_type}"},
                status=status.HTTP_422_UNPROCESSABLE_ENTITY,
            )

        # Idempotency: check for existing fragment
        try:
            existing = Fragment.objects.get(fragment_id=fragment_id)
            if existing.status == Fragment.Status.COMPLETE:
                return Response({
                    "fragment_id": str(existing.fragment_id),
                    "status": "complete",
                    "model_version": existing.model_version,
                    **existing.analysis,
                })
            if existing.status in (Fragment.Status.PENDING, Fragment.Status.ANALYZING):
                return Response(
                    {"fragment_id": str(existing.fragment_id),
                     "status": "pending",
                     "poll_url": f"/api/analyse/{existing.fragment_id}/"},
                    status=status.HTTP_202_ACCEPTED,
                )
            if existing.status == Fragment.Status.FAILED:
                # Re-queue failed fragments
                existing.status = Fragment.Status.PENDING
                existing.error_message = ""
                existing.save(update_fields=["status", "error_message"])
                try:
                    analyse_fragment.delay(str(fragment_id))
                except Exception:
                    existing.status = Fragment.Status.FAILED
                    existing.error_message = "Could not queue analysis"
                    existing.save(update_fields=["status", "error_message"])
                    return Response(
                        {"error": "service_unavailable", "message": "Analysis service unavailable"},
                        status=status.HTTP_503_SERVICE_UNAVAILABLE,
                    )
                return Response(
                    {"fragment_id": str(existing.fragment_id),
                     "status": "pending",
                     "poll_url": f"/api/analyse/{existing.fragment_id}/"},
                    status=status.HTTP_202_ACCEPTED,
                )
        except Fragment.DoesNotExist:
            pass

        # Save audio file
        filename = f"fragments/{fragment_id}{_ext_from_mime(mime_type)}"
        saved_path = default_storage.save(filename, audio)

        # Create fragment
        model_version = getattr(settings, "SKETCHBOOK_MODEL_VERSION", "0.0.0")
        fragment = Fragment.objects.create(
            fragment_id=fragment_id,
            user=request.user,
            audio_storage_path=saved_path,
            mime_type=mime_type,
            duration_seconds=duration_seconds,
            model_version=model_version,
        )

        # Queue analysis
        try:
            analyse_fragment.delay(str(fragment_id))
        except Exception:
            fragment.status = Fragment.Status.FAILED
            fragment.error_message = "Could not queue analysis"
            fragment.save(update_fields=["status", "error_message"])
            return Response(
                {"error": "service_unavailable", "message": "Analysis service unavailable"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        return Response(
            {"fragment_id": str(fragment.fragment_id),
             "status": "pending",
             "poll_url": f"/api/analyse/{fragment.fragment_id}/"},
            status=status.HTTP_202_ACCEPTED,
        )

    def get(self, request, fragment_id):
        try:
            fragment_uuid = uuid.UUID(str(fragment_id))
            fragment = Fragment.objects.get(fragment_id=fragment_uuid)
        except (ValueError, Fragment.DoesNotExist):
            return Response(
                {"detail": "Not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        if fragment.status == Fragment.Status.COMPLETE:
            return Response({
                "fragment_id": str(fragment.fragment_id),
                "status": "complete",
                "model_version": fragment.model_version,
                **fragment.analysis,
            })

        if fragment.status == Fragment.Status.FAILED:
            return Response({
                "fragment_id": str(fragment.fragment_id),
                "status": "failed",
                "error": fragment.error_message,
            })

        # PENDING or ANALYZING
        return Response({
            "fragment_id": str(fragment.fragment_id),
            "status": "pending",
        })


def _ext_from_mime(mime_type: str) -> str:
    """Map mime type to file extension."""
    mapping = {
        "audio/wav": ".wav", "audio/wave": ".wav", "audio/x-wav": ".wav",
        "audio/mp3": ".mp3", "audio/mpeg": ".mp3",
        "audio/flac": ".flac", "audio/ogg": ".ogg",
        "audio/mp4": ".mp4", "audio/m4a": ".m4a", "audio/x-m4a": ".m4a",
        "audio/webm": ".webm", "audio/aac": ".aac",
    }
    return mapping.get(mime_type, ".bin")
```

```python
# src/sketchbook/urls.py
"""URL routing for the Musical Sketchbook API."""

from django.urls import path
from . import views

urlpatterns = [
    path("health/", views.HealthView.as_view(), name="sketchbook-health"),
    path("analyse/", views.AnalyseView.as_view(), name="sketchbook-analyse"),
    path("analyse/<uuid:fragment_id>/", views.AnalyseView.as_view(), name="sketchbook-analyse-detail"),
]
```

Update `config/urls.py` to include sketchbook URLs:

```python
"""URL configuration for audio-analysis-agents project."""

from django.conf import settings
from django.conf.urls.static import static
from django.urls import path, include
from django.views.generic import TemplateView

urlpatterns = [
    path("", TemplateView.as_view(template_name="index.html")),
    path("api/", include("src.api.urls")),
    path("api/", include("src.sketchbook.urls")),
]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATICFILES_DIRS[0])
```

Update `tests/conftest.py` to include `rest_framework` and `REST_FRAMEWORK` settings:

```python
import django
from django.conf import settings


def pytest_configure():
    """Configure Django settings before tests."""
    if not settings.configured:
        settings.configure(
            DEBUG=True,
            SECRET_KEY="test-secret-key-for-testing-only",
            DATABASES={
                "default": {
                    "ENGINE": "django.db.backends.sqlite3",
                    "NAME": ":memory:",
                }
            },
            INSTALLED_APPS=[
                "django.contrib.auth",
                "django.contrib.contenttypes",
                "rest_framework",
                "src.api",
                "src.sketchbook",
            ],
            DEFAULT_AUTO_FIELD="django.db.models.BigAutoField",
            USE_TZ=True,
            ROOT_URLCONF="config.urls",
            SKETCHBOOK_MODEL_VERSION="1.0.0",
            REST_FRAMEWORK={
                "DEFAULT_AUTHENTICATION_CLASSES": [
                    "rest_framework_simplejwt.authentication.JWTAuthentication",
                ],
                "DEFAULT_PERMISSION_CLASSES": [],
                "UNAUTHENTICATED_USER": None,
            },
            MEDIA_ROOT="/tmp/test_media",
        )
    django.setup()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_sketchbook/test_views.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/sketchbook/views.py src/sketchbook/urls.py config/urls.py tests/conftest.py tests/test_sketchbook/test_views.py
git commit -m "feat(sketchbook): add API views and URL routing"
```

---

### Task 10: Run full test suite

**Step 1: Run all tests**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS

**Step 2: Fix any failures**

If tests fail, identify the root cause and fix. Common issues:
- conftest.py changes may break existing tests (check INSTALLED_APPS, REST_FRAMEWORK)
- Import errors from circular dependencies

**Step 3: Commit any fixes**

```bash
git add -A
git commit -m "fix: resolve test failures from sketchbook integration"
```

---

### Task 11: Create Django migration

**Step 1: Generate migration**

Run: `python manage.py makemigrations sketchbook`
Expected: Creates a migration file in `src/sketchbook/migrations/`

**Step 2: Verify migration applies**

Run: `python manage.py migrate`
Expected: Migration applies without errors

**Step 3: Commit**

```bash
git add src/sketchbook/migrations/
git commit -m "chore(sketchbook): add initial Fragment migration"
```

---

### Task 12: End-to-end integration test

**Files:**
- Test: `tests/test_sketchbook/test_integration.py`

**Step 1: Write the integration test**

```python
# tests/test_sketchbook/test_integration.py
"""End-to-end integration test for the sketchbook API flow."""

import uuid
import pytest
from unittest.mock import patch, MagicMock
from django.test import TestCase
from django.contrib.auth.models import User
from rest_framework.test import APIClient
from src.sketchbook.models import Fragment
from src.agents.base import AnalysisResult


class TestAnalyseEndToEnd(TestCase):
    """Test the full POST → task → GET polling flow."""

    def setUp(self):
        self.user = User.objects.create_user(username="test", password="test")
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)

    @patch("src.sketchbook.views.analyse_fragment")
    def test_post_then_task_then_get(self, mock_task):
        """POST creates fragment, simulated task completes it, GET returns result."""
        mock_task.delay.return_value = MagicMock(id="task-1")
        fid = uuid.uuid4()

        from django.core.files.uploadedfile import SimpleUploadedFile
        audio = SimpleUploadedFile("test.wav", b"data", content_type="audio/wav")

        # POST — should return 202
        post_resp = self.client.post(
            "/api/analyse/",
            {"audio": audio, "mime_type": "audio/wav",
             "fragment_id": str(fid), "duration_seconds": "5.0"},
            format="multipart",
        )
        assert post_resp.status_code == 202

        # GET while pending — should return pending
        get_resp = self.client.get(f"/api/analyse/{fid}/")
        assert get_resp.status_code == 200
        assert get_resp.json()["status"] == "pending"

        # Simulate task completing
        fragment = Fragment.objects.get(fragment_id=fid)
        fragment.status = Fragment.Status.COMPLETE
        fragment.analysis = {
            "dimensions": {"bpm": 120, "time_signature": "4/4",
                          "swing": 0.1, "steadiness": 0.9, "upbeat": False},
            "descriptors": ["moderate-tempo", "straight", "steady"],
            "raw_data": {"beats": [0.5], "onsets": [0.1],
                        "spectral_centroid_mean": 2000, "rms_energy_mean": 0.05},
        }
        fragment.save()

        # GET after completion — should return full result
        get_resp = self.client.get(f"/api/analyse/{fid}/")
        assert get_resp.status_code == 200
        data = get_resp.json()
        assert data["status"] == "complete"
        assert data["dimensions"]["bpm"] == 120
        assert "descriptors" in data
        assert "raw_data" in data

        # Re-POST same fragment_id — should return 200 with cached result
        audio2 = SimpleUploadedFile("test.wav", b"data", content_type="audio/wav")
        repost_resp = self.client.post(
            "/api/analyse/",
            {"audio": audio2, "mime_type": "audio/wav",
             "fragment_id": str(fid), "duration_seconds": "5.0"},
            format="multipart",
        )
        assert repost_resp.status_code == 200
        assert repost_resp.json()["dimensions"]["bpm"] == 120
```

**Step 2: Run the integration test**

Run: `pytest tests/test_sketchbook/test_integration.py -v`
Expected: PASS

**Step 3: Run full suite one more time**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add tests/test_sketchbook/test_integration.py
git commit -m "test(sketchbook): add end-to-end integration test"
```
