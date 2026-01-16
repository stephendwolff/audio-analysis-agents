# Audio Analysis Agents

A modular system for analysing sound files using specialised tools coordinated by an orchestrator.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Orchestrator                            │
│  - Receives analysis requests                                │
│  - Plans which tools/agents to invoke                        │
│  - Aggregates results                                        │
└─────────────────┬───────────────────────────────────────────┘
                  │
    ┌─────────────┼─────────────┬─────────────┐
    ▼             ▼             ▼             ▼
┌────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│Spectral│  │ Temporal │  │  Rhythm  │  │  Custom  │
│Analysis│  │ Analysis │  │ Analysis │  │  Agent   │
└────────┘  └──────────┘  └──────────┘  └──────────┘
```

## Project Structure

```
audio-analysis-agents/
├── src/
│   ├── orchestrator/       # Main coordinator
│   │   ├── __init__.py
│   │   └── orchestrator.py
│   ├── agents/             # Specialised analysis agents
│   │   ├── __init__.py
│   │   ├── base.py         # Base agent class
│   │   ├── spectral.py     # FFT, spectrograms, frequency analysis
│   │   ├── temporal.py     # Waveform, envelope, dynamics
│   │   └── rhythm.py       # BPM, onset detection, beats
│   └── tools/              # Low-level audio utilities
│       ├── __init__.py
│       ├── loader.py       # Audio file loading
│       └── features.py     # Feature extraction helpers
├── tests/
├── data/
│   ├── samples/            # Test audio files
│   └── outputs/            # Analysis results
├── docs/
├── pyproject.toml
└── README.md
```

## Installation

```bash
cd audio-analysis-agents
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## Quick Start

```python
from src.orchestrator import Orchestrator

orchestrator = Orchestrator()

# Analyse a sound file
results = orchestrator.analyse("path/to/audio.wav")

# Or run specific analyses
results = orchestrator.analyse(
    "path/to/audio.wav",
    tasks=["spectral", "rhythm"]
)
```

## Available Agents

| Agent | Description | Key Outputs |
|-------|-------------|-------------|
| `spectral` | Frequency domain analysis | Spectrogram, dominant frequencies, spectral centroid |
| `temporal` | Time domain analysis | Waveform stats, RMS energy, zero-crossing rate |
| `rhythm` | Rhythmic analysis | BPM, beat positions, onset times |

## Adding Custom Agents

```python
from src.agents.base import BaseAgent, AnalysisResult

class MyCustomAgent(BaseAgent):
    name = "custom"
    description = "My custom analysis"

    def analyse(self, audio_data, sample_rate) -> AnalysisResult:
        # Your analysis logic here
        return AnalysisResult(
            agent=self.name,
            success=True,
            data={"my_metric": 42}
        )
```

## Dependencies

- `librosa` - Audio analysis
- `numpy` - Numerical operations
- `soundfile` - Audio file I/O
- `pydantic` - Data validation

## License

MIT
