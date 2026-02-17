# Audio Analysis Agents

A modular system for analysing audio using specialised agents. Use the agents directly as a Python library, or run the Django API to serve analysis over HTTP.

## Architecture

```
                    ┌──────────────────────────────────┐
                    │         Analysis Agents           │
                    │  spectral  ·  temporal  ·  rhythm │
                    └──────────┬───────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                                 ▼
     ┌─────────────────┐              ┌─────────────────┐
     │  Python Library  │              │    HTTP API     │
     │  Import & call   │              │  Django + DRF   │
     │  agents directly │              │  Celery workers │
     └─────────────────┘              └─────────────────┘
```

**Library mode** -- import agents, pass numpy arrays, get structured results. No Django required.

**API mode** -- upload audio over HTTP, poll for results. Handles auth, storage, background processing.

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
# Core (agents only)
uv sync

# With API server
uv sync --extra api --extra dev

# With LLM routing
uv sync --extra litellm
```

The API also requires ffmpeg (for audio format conversion) and Redis (for Celery):

```bash
# macOS
brew install ffmpeg redis
```

## Quick Start: Library

```python
from src.agents import RhythmAgent, SpectralAgent, TemporalAgent
from src.tools.loader import load_audio

audio = load_audio("path/to/audio.wav")

rhythm = RhythmAgent().analyse(audio.samples, audio.sample_rate)
print(rhythm.data["tempo_bpm"])       # 120.5
print(rhythm.data["time_signature"])  # "4/4"
print(rhythm.data["steadiness"])      # 0.87
```

See [docs/agents.md](docs/agents.md) for full agent reference.

## Quick Start: API

```bash
# Start the server
python manage.py migrate
python manage.py runserver

# Submit audio for analysis
curl -X POST http://localhost:8000/api/analyse/ \
  -H "Authorization: Bearer <jwt>" \
  -F "audio=@recording.wav" \
  -F "mime_type=audio/wav" \
  -F "fragment_id=550e8400-e29b-41d4-a716-446655440000" \
  -F "duration_seconds=5.0"

# Poll for results
curl http://localhost:8000/api/analyse/550e8400-e29b-41d4-a716-446655440000/ \
  -H "Authorization: Bearer <jwt>"
```

See [docs/api.md](docs/api.md) for full endpoint reference.

## Documentation

| Document | Contents |
|----------|----------|
| [docs/architecture.md](docs/architecture.md) | System design, components, planned features |
| [docs/agents.md](docs/agents.md) | Agent library: built-in agents, outputs, custom agents |
| [docs/api.md](docs/api.md) | HTTP API: endpoints, auth, request/response formats |
| [docs/routing.md](docs/routing.md) | LLM-powered routing: orchestrator, provider comparison |

## Dependencies

**Core:** librosa, numpy, soundfile, pydantic

**API:** Django, DRF, Celery, Redis, PostgreSQL, S3 (optional)

**LLM routing:** LiteLLM (multi-provider) or google-genai (Gemini-only) or LangGraph

See `pyproject.toml` for the full dependency list grouped by feature.

## License

MIT
