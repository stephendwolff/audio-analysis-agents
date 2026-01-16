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

## Railway Deployment

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DJANGO_SECRET_KEY` | Yes | Generate a secure random key (50+ chars) |
| `DJANGO_DEBUG` | Yes | Set to `false` for production |
| `DJANGO_ALLOWED_HOSTS` | Yes | Set to `.railway.app` (or your custom domain) |
| `DATABASE_URL` | Auto | Provided automatically by Railway PostgreSQL |
| `MEDIA_ROOT` | Yes | Set to `/app/data/uploads` (mount volume here) |
| `API_KEY` | Yes | Your API key for authentication |
| `LLM_MODEL` | No | LLM model to use (default: `gemini/gemini-2.0-flash`) |
| `GOOGLE_API_KEY` | Conditional | Required if using Gemini models |
| `ANTHROPIC_API_KEY` | Conditional | Required if using Claude models |

### Setup Steps

#### 1. Create Railway Project

- Go to [railway.app](https://railway.app) and create a new project
- Select "Deploy from GitHub repo" and connect your repository

#### 2. Add PostgreSQL Database

- In your project, click **+ New** → **Database** → **Add PostgreSQL**
- Once created, click on the PostgreSQL service
- Go to **Variables** tab and copy `DATABASE_URL`
- Go to your app service → **Variables** → **New Variable**
- Click **Add Reference** → select the PostgreSQL service → select `DATABASE_URL`
- This automatically links the database URL to your app

#### 3. Add Volume for Media Files

- Click on your app service → **Settings** tab
- Scroll to **Volumes** section → **+ Add Volume**
- Set mount path: `/app/data/uploads`
- Click **Add**
- Go to **Variables** tab and add: `MEDIA_ROOT` = `/app/data/uploads`

#### 4. Set Environment Variables

In your app service → **Variables** tab, add:

```
DJANGO_SECRET_KEY=<generate with: python -c "import secrets; print(secrets.token_urlsafe(50))">
DJANGO_DEBUG=false
DJANGO_ALLOWED_HOSTS=.railway.app
MEDIA_ROOT=/app/data/uploads
API_KEY=<your-api-key>
GOOGLE_API_KEY=<your-google-api-key>
```

## License

MIT
