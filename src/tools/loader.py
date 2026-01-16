"""Audio file loading utilities."""

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


def load_audio(
    file_path: str | Path,
    target_sr: Optional[int] = None,
    mono: bool = True,
) -> AudioData:
    """
    Load an audio file.

    Args:
        file_path: Path to the audio file
        target_sr: Target sample rate (None to keep original)
        mono: Convert to mono if True

    Returns:
        AudioData containing the loaded audio
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # Load with soundfile
    samples, sample_rate = sf.read(file_path, always_2d=True)

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
        file_path=file_path,
    )
