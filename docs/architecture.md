# Architecture

## Overview

Audio Analysis Agents has two layers: a Python library of analysis agents, and a Django API that serves those agents over HTTP.

```
┌─────────────────────────────────────────────────────────────┐
│                       HTTP API Layer                        │
│                                                             │
│  src/api/          Auth, tracks, chat, WebSocket            │
│  src/sketchbook/   Fragment analysis (async via Celery)     │
│  config/           Django settings, URLs, ASGI              │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────┴───────────────────────────────────┐
│                      Core Library                           │
│                                                             │
│  src/agents/       Spectral, Temporal, Rhythm agents        │
│  src/orchestrator/  LLM routing (Gemini, LiteLLM, etc.)    │
│  src/tools/        Audio loading, feature extraction        │
│  src/observability/ Opik tracing                            │
└─────────────────────────────────────────────────────────────┘
```

The core library has no Django dependency. Agents take numpy arrays and return structured results. The API layer adds authentication, storage, and background processing on top.

## Components

### Analysis Agents (`src/agents/`)

Standalone Python classes that analyse audio. Each agent:
- Takes `(samples: np.ndarray, sample_rate: int)`
- Returns an `AnalysisResult` with structured data
- Registers itself via the `@register_agent` decorator

Built-in agents: **spectral**, **temporal**, **rhythm**. See [agents.md](agents.md).

### Orchestrator (`src/orchestrator/`)

Connects natural language questions to the right agents via LLM tool calling. Three implementations exist, each using a different LLM integration approach:

- **LLMRouter** -- Gemini SDK directly
- **LLMRouterLiteLLM** -- any provider via LiteLLM
- **LLMRouterLangGraph** -- LangGraph state machine

See [routing.md](routing.md).

### Sketchbook API (`src/sketchbook/`)

The Musical Sketchbook endpoints. An iOS app submits audio fragments; the server analyses them asynchronously via Celery and returns structured results.

- `POST /api/analyse/` -- submit audio, get 202 with poll URL
- `GET /api/analyse/{fragment_id}/` -- poll for results
- `GET /api/health/` -- server availability

See [api.md](api.md).

### General API (`src/api/`)

Authentication, track management, and WebSocket support for the chat-based analysis interface.

- JWT auth via `djangorestframework-simplejwt`
- API key auth (legacy)
- Demo tokens with rate limiting
- WebSocket consumer for streaming LLM responses

### Tools (`src/tools/`)

Low-level audio utilities.

- `loader.py` -- load audio files (wav, mp3, flac, ogg, mp4, m4a, webm, aac), convert to wav via pydub, resample
- `features.py` -- shared feature extraction helpers

### Observability (`src/observability/`)

Opik tracing for LLM calls and tool executions. Enabled via `OPIK_API_KEY` environment variable.

## Project Structure

```
audio-analysis-agents/
├── config/                  Django settings, URLs, ASGI/WSGI
├── src/
│   ├── agents/              Analysis agents (no Django dependency)
│   │   ├── base.py          BaseAgent, AnalysisResult
│   │   ├── registry.py      Agent registration and discovery
│   │   ├── spectral.py      Frequency analysis
│   │   ├── temporal.py      Time-domain analysis
│   │   └── rhythm.py        Beat, tempo, time signature
│   ├── orchestrator/        LLM routing
│   │   ├── orchestrator.py  Rule-based orchestrator
│   │   ├── llm_routing.py   Gemini SDK router
│   │   ├── llm_routing_litellm.py   LiteLLM router
│   │   └── llm_routing_langgraph.py LangGraph router
│   ├── tools/               Audio loading and feature extraction
│   ├── api/                 Django app: auth, tracks, WebSocket
│   ├── sketchbook/          Django app: fragment analysis API
│   ├── tasks/               Celery task definitions
│   └── observability/       Opik tracing
├── tests/
├── evals/                   LLM routing evaluation suite
├── examples/                Usage examples
├── docs/
│   ├── architecture.md      This file
│   ├── agents.md            Agent reference
│   ├── api.md               HTTP API reference
│   ├── routing.md           LLM routing reference
│   └── plans/               Implementation plans
└── pyproject.toml
```

## Deployment

Four Railway services, all from the same repo:

| Service | Role | Start command |
|---------|------|---------------|
| **web** | Django ASGI server | `python manage.py migrate --noinput && gunicorn config.asgi:application -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT` |
| **audio-analysis-agents** | Celery worker | `celery -A src.tasks.celery worker --loglevel=info` |
| **Postgres** | Database | Managed by Railway |
| **Redis** | Celery broker + channel layers | Managed by Railway |

The build is configured in `nixpacks.toml` (shared by web and worker). Start commands are set per-service in the Railway dashboard. Audio storage uses S3 in production or a Railway volume for dev.

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DJANGO_SECRET_KEY` | Yes | Secure random key |
| `DJANGO_DEBUG` | Yes | `false` in production |
| `DJANGO_ALLOWED_HOSTS` | Yes | `.railway.app` or custom domain |
| `DATABASE_URL` | Auto | Provided by Railway PostgreSQL |
| `CELERY_BROKER_URL` | Yes | Redis URL for Celery |
| `LLM_MODEL` | No | Default: `gemini/gemini-2.0-flash` |
| `GOOGLE_API_KEY` | Conditional | Required for Gemini models |
| `ANTHROPIC_API_KEY` | Conditional | Required for Claude models |
| `AWS_STORAGE_BUCKET_NAME` | No | Enables S3 storage |
| `AWS_ACCESS_KEY_ID` | Conditional | Required with S3 |
| `AWS_SECRET_ACCESS_KEY` | Conditional | Required with S3 |
| `SENTRY_DSN` | No | Enables Sentry error tracking |
| `SENTRY_ENVIRONMENT` | No | Default: `production` |

## Planned

Features designed but not yet built:

- **Track management API** -- upload, list, delete audio tracks with metadata; cached analysis results per track
- **Chat interface** -- natural language questions about uploaded audio, LLM-generated responses using agent results
- **MCP server** -- expose analysis tools to external LLM clients via Model Context Protocol (HTTP transport)
- **VAMPNET integration** -- audio manipulation and generation
- **Comparison tools** -- analyse differences between tracks
- **Visualisation endpoints** -- return spectrograms and waveforms as images
- **Analysis summaries** -- LLM-generated natural language descriptions of sketchbook analysis results (e.g. "A laid-back, swung groove at 92 BPM with a steady pulse"); would bring Opik tracing to the sketchbook API path
- **Streaming analysis** -- progressive results for long files
