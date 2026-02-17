# Agents

Analysis agents are standalone Python classes with no Django dependency. Each agent takes numpy audio data and returns structured results.

## Built-in Agents

| Agent | Key Outputs |
|-------|-------------|
| **spectral** | Spectral centroid, bandwidth, rolloff, contrast, MFCCs, dominant frequencies, flatness |
| **temporal** | RMS energy, zero-crossing rate, amplitude envelope, duration, dynamics |
| **rhythm** | BPM, beat times, onset times, tempo stability, time signature, swing, steadiness, upbeat |

## Usage

```python
from src.agents import RhythmAgent, SpectralAgent, TemporalAgent
from src.tools.loader import load_audio

audio = load_audio("track.wav", target_sr=22050, mono=True)

rhythm = RhythmAgent().analyse(audio.samples, audio.sample_rate)
spectral = SpectralAgent().analyse(audio.samples, audio.sample_rate)
temporal = TemporalAgent().analyse(audio.samples, audio.sample_rate)
```

### Loading Audio

`load_audio` returns an `AudioData` object with `.samples` (numpy array) and `.sample_rate`.

It supports wav, mp3, flac, ogg, mp4, m4a, webm, and aac. Non-wav formats are converted via pydub/ffmpeg.

```python
from src.tools.loader import load_audio

# From file path
audio = load_audio("path/to/file.m4a", mime_type="audio/m4a")

# From BytesIO
import io
audio = load_audio(io.BytesIO(raw_bytes), mime_type="audio/webm")
```

### AnalysisResult

Every agent returns an `AnalysisResult`:

```python
class AnalysisResult(BaseModel):
    agent: str              # Agent name ("rhythm", "spectral", etc.)
    success: bool           # Whether analysis succeeded
    data: dict[str, Any]    # Analysis output
    error: str | None       # Error message if failed
```

Check `result.success` before accessing `result.data`.

## Rhythm Agent Output

```python
{
    "tempo_bpm": 120.5,
    "beat_count": 48,
    "beat_times": [0.52, 1.04, ...],          # All beat positions (seconds)
    "onset_count": 96,
    "onset_times": [0.01, 0.26, ...],         # All onset positions (seconds)
    "tempo_stability": {
        "mean_ibi": 0.498,                    # Mean inter-beat interval
        "std_ibi": 0.012,
        "cv": 0.024                           # Coefficient of variation
    },
    "onset_strength": {"mean": 5.2, "max": 18.3, "std": 3.1},
    "tempo_range": {"dominant_tempos": [120.5, 60.2]},
    "time_signature": "4/4",                  # Detected time signature
    "swing": 0.15,                            # 0.0 (straight) to 1.0 (full triplet)
    "steadiness": 0.87,                       # 0.0 (loose) to 1.0 (metronomic)
    "upbeat": false                           # Anacrusis detected
}
```

## Spectral Agent Output

```python
{
    "spectral_centroid": {"mean": 2400.5, "std": 800.1},
    "spectral_bandwidth": {"mean": 2100.3, "std": 500.2},
    "spectral_rolloff": {"mean": 5200.0, "std": 1200.0},
    "spectral_contrast": {"bands": [...]},
    "spectral_flatness": {"mean": 0.02, "std": 0.01},
    "mfccs": {"means": [...], "stds": [...]},
    "dominant_frequencies": [440.0, 880.0, 1320.0],
    "frequency_distribution": {"low": 0.3, "mid": 0.5, "high": 0.2}
}
```

## Temporal Agent Output

```python
{
    "duration_seconds": 30.5,
    "rms_energy": {"mean": 0.034, "std": 0.01, "max": 0.12},
    "zero_crossing_rate": {"mean": 0.08, "std": 0.02},
    "amplitude_envelope": {"peak": 0.95, "mean": 0.12},
    "dynamics": {"range_db": 24.5, "crest_factor": 8.2}
}
```

## Agent Registry

Agents register themselves via the `@register_agent` decorator. The registry provides discovery and OpenAI-format tool schemas for LLM integration.

```python
from src.agents import get_all_agents, get_agent, get_tool_schemas_openai

# List all registered agents
agents = get_all_agents()  # [SpectralAgent(), TemporalAgent(), RhythmAgent()]

# Get a specific agent
rhythm = get_agent("rhythm")

# Generate OpenAI tool schemas (for LLM routing)
schemas = get_tool_schemas_openai()
```

## Custom Agents

Subclass `BaseAgent`, set `name` and `description`, implement `analyse`, and decorate with `@register_agent`:

```python
from src.agents.base import BaseAgent, AnalysisResult
from src.agents.registry import register_agent

@register_agent
class HarmonyAgent(BaseAgent):
    name = "harmony"
    description = "Analyse harmonic content. Returns key, chords, and harmonic complexity."

    def analyse(self, samples, sample_rate):
        try:
            import librosa
            chroma = librosa.feature.chroma_cqt(y=samples, sr=sample_rate)
            # ... analysis logic ...
            return AnalysisResult(agent=self.name, success=True, data={...})
        except Exception as e:
            return AnalysisResult(agent=self.name, success=False, error=str(e))
```

The `description` field matters -- it is used by the LLM router to decide which agent to invoke. Write it as if explaining the tool to an LLM.

Once registered, the agent is automatically available to the orchestrator, the API health endpoint, and `get_all_agents()`.
