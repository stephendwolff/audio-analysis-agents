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
    """Convert an audio container format to WAV using pydub/ffmpeg.

    Args:
        file_obj: BytesIO containing the audio data
        mime_type: MIME type of the audio data

    Returns:
        BytesIO containing the WAV audio data

    Raises:
        ValueError: If the mime_type is not supported for conversion
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
    file_path: str | Path | io.BytesIO,
    target_sr: Optional[int] = None,
    mono: bool = True,
    mime_type: Optional[str] = None,
) -> AudioData:
    """
    Load an audio file from a local path, URL, or in-memory BytesIO buffer.

    Args:
        file_path: Path to the audio file, URL, or BytesIO buffer
        target_sr: Target sample rate (None to keep original)
        mono: Convert to mono if True
        mime_type: MIME type of the audio data (used for container format conversion)

    Returns:
        AudioData containing the loaded audio
    """
    # Handle in-memory BytesIO input
    if isinstance(file_path, io.BytesIO):
        if mime_type and mime_type in MIME_TO_FORMAT:
            file_path = convert_to_wav(file_path, mime_type)
        samples, sample_rate = sf.read(file_path, always_2d=True)
        source_path = None
    else:
        file_path_str = str(file_path)

        if _is_url(file_path_str):
            # Download from URL and load from memory
            try:
                buffer = _download_to_buffer(file_path_str)
                if mime_type and mime_type in MIME_TO_FORMAT:
                    buffer = convert_to_wav(buffer, mime_type)
                samples, sample_rate = sf.read(buffer, always_2d=True)
            except FileNotFoundError:
                raise
            except Exception as e:
                raise FileNotFoundError(f"Could not load audio from URL: {file_path_str}. Error: {e}")
            source_path = None
        else:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Audio file not found: {file_path}")
            if mime_type and mime_type in MIME_TO_FORMAT:
                with open(file_path, "rb") as f:
                    buffer = io.BytesIO(f.read())
                buffer = convert_to_wav(buffer, mime_type)
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
