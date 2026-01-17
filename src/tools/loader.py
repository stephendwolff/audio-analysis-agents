"""Audio file loading utilities."""

import io
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf


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


def load_audio(
    file_path: str | Path,
    target_sr: Optional[int] = None,
    mono: bool = True,
) -> AudioData:
    """
    Load an audio file from a local path or URL.

    Args:
        file_path: Path to the audio file or URL
        target_sr: Target sample rate (None to keep original)
        mono: Convert to mono if True

    Returns:
        AudioData containing the loaded audio
    """
    file_path_str = str(file_path)

    if _is_url(file_path_str):
        # Download from URL and load from memory
        try:
            buffer = _download_to_buffer(file_path_str)
            samples, sample_rate = sf.read(buffer, always_2d=True)
        except Exception as e:
            raise FileNotFoundError(f"Could not load audio from URL: {file_path_str}. Error: {e}")
        source_path = None
    else:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
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
