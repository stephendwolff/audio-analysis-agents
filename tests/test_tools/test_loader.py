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
    def test_converts_mp4_to_wav_bytes(self):
        fake_input = io.BytesIO(b"fake-mp4-data")
        wav_bytes = _make_wav_bytes()
        mock_segment = MagicMock()
        mock_segment.export = MagicMock(side_effect=lambda buf, format: buf.write(wav_bytes))
        mock_audio_segment = MagicMock()
        mock_audio_segment.from_file.return_value = mock_segment
        mock_pydub = MagicMock()
        mock_pydub.AudioSegment = mock_audio_segment
        with patch.dict("sys.modules", {"pydub": mock_pydub}):
            result = convert_to_wav(fake_input, "audio/mp4")
        assert isinstance(result, io.BytesIO)
        mock_audio_segment.from_file.assert_called_once()

    def test_raises_on_unsupported_mime(self):
        with pytest.raises(ValueError, match="Unsupported"):
            convert_to_wav(io.BytesIO(b"data"), "audio/xyz")


class TestLoadAudioWithMimeType:
    def test_load_audio_with_mime_type_converts_first(self):
        wav_bytes = _make_wav_bytes()
        with patch("src.tools.loader.convert_to_wav") as mock_convert:
            mock_convert.return_value = io.BytesIO(wav_bytes)
            result = load_audio(io.BytesIO(b"fake"), mime_type="audio/mp4")
        mock_convert.assert_called_once()
        assert result.sample_rate > 0

    def test_load_audio_without_mime_type_skips_conversion(self):
        wav_bytes = _make_wav_bytes()
        with patch("src.tools.loader.convert_to_wav") as mock_convert:
            result = load_audio(io.BytesIO(wav_bytes))
        mock_convert.assert_not_called()
