# Audio Analysis Agents - Design Document

## Overview

A Django-based system for analysing audio files via LLM-accessible tools. Users upload audio through a frontend, ask natural language questions, and receive responses powered by specialised analysis agents.

## Architecture

```
┌─────────────┐     ┌─────────────────────────────────────────────────┐
│   Frontend  │     │           Audio Analysis Project                │
│             │     │                                                 │
│  - Upload   │────▶│  ┌─────────┐    ┌──────────────┐               │
│  - Chat UI  │     │  │ Django  │    │ Orchestrator │               │
│             │◀────│  │   API   │───▶│              │               │
└─────────────┘     │  └─────────┘    └──────┬───────┘               │
                    │                        │                        │
                    │         ┌──────────────┼──────────────┐        │
                    │         ▼              ▼              ▼        │
                    │    ┌────────┐    ┌──────────┐   ┌────────┐    │
                    │    │Spectral│    │ Temporal │   │ Rhythm │    │
                    │    │ Agent  │    │  Agent   │   │ Agent  │    │
                    │    └────────┘    └──────────┘   └────────┘    │
                    │                                                 │
                    │    ┌─────────────────────────────────────┐     │
                    │    │           MCP Server (HTTP)          │     │
                    │    │  - Exposes tools to external LLMs    │     │
                    │    └─────────────────────────────────────┘     │
                    └─────────────────────────────────────────────────┘

                    ┌─────────────┐
                    │ LLM Service │  (Claude, Gemini, ChatGPT)
                    │             │◀──── MCP tool calls
                    └─────────────┘
```

## User Flow

1. User uploads audio file via frontend
2. Frontend POSTs to Django API → file stored, Track model created
3. User asks question in chat (e.g. "what tempo is this?")
4. Frontend sends question + track reference to API
5. Orchestrator decides which tools to invoke (rule-based OR LLM-assisted)
6. Analysis runs, results passed to LLM
7. LLM formulates natural language response
8. Response returned to frontend

## Components

### 1. Django API (DRF)

**Models:**
```python
class Track(models.Model):
    id = models.UUIDField(primary_key=True)
    name = models.CharField(max_length=255)
    file = models.FileField(upload_to='tracks/')
    duration = models.FloatField(null=True)
    sample_rate = models.IntegerField(null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    metadata = models.JSONField(default=dict)  # extensible

class AnalysisResult(models.Model):
    track = models.ForeignKey(Track, on_delete=models.CASCADE)
    agent = models.CharField(max_length=50)  # 'spectral', 'rhythm', etc.
    result = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
```

**Endpoints:**
```
POST   /api/tracks/              # Upload audio file
GET    /api/tracks/{id}/         # Get track info
DELETE /api/tracks/{id}/         # Remove track

POST   /api/tracks/{id}/analyse/ # Run analysis (optional: specify agents)
GET    /api/tracks/{id}/results/ # Get cached analysis results

POST   /api/chat/                # Send question, get LLM response
```

### 2. Analysis Agents

Standalone Python classes, no Django dependency. Each agent:
- Takes numpy samples + sample rate
- Returns structured `AnalysisResult`

**Built-in agents:**
| Agent | Purpose | Key Outputs |
|-------|---------|-------------|
| `spectral` | Frequency analysis | FFT, spectrogram, MFCCs, dominant frequencies |
| `temporal` | Time-domain analysis | Duration, RMS, amplitude envelope, dynamics |
| `rhythm` | Beat/tempo analysis | BPM, beat positions, onsets, tempo stability |

**Extensibility:**
```python
from agents.base import BaseAgent, AnalysisResult

class MyAgent(BaseAgent):
    name = "my_agent"
    description = "Does something custom"

    def analyse(self, samples, sample_rate) -> AnalysisResult:
        # ...
```

### 3. Orchestrator

Sits between API and agents. Two routing modes:

**A. Rule-based routing:**
- Keyword/intent matching on user question
- Maps to appropriate agents
- Fast, predictable, no LLM cost

**B. LLM-assisted routing:**
- Sends question to cheap/fast model
- Model decides which tools to call
- More flexible for complex questions

Config flag to switch modes, or hybrid approach.

```python
class Orchestrator:
    def __init__(self, routing_mode='rules'):  # or 'llm' or 'hybrid'
        ...

    def process_question(self, track_id: str, question: str) -> dict:
        # 1. Determine which agents to run
        # 2. Run analyses (or fetch cached)
        # 3. Pass results + question to LLM
        # 4. Return response
```

### 4. MCP Server (HTTP)

Exposes analysis tools to external LLM clients.

**Transport:** HTTP (not stdio) - required for network access from frontend LLM

**Tools exposed:**
```json
{
  "tools": [
    {
      "name": "analyse_spectral",
      "description": "Analyse frequency content of audio",
      "parameters": {
        "track_id": "string"
      }
    },
    {
      "name": "analyse_rhythm",
      "description": "Analyse tempo and beats",
      "parameters": {
        "track_id": "string"
      }
    },
    {
      "name": "analyse_temporal",
      "description": "Analyse time-domain properties",
      "parameters": {
        "track_id": "string"
      }
    },
    {
      "name": "get_track_info",
      "description": "Get basic info about uploaded track",
      "parameters": {
        "track_id": "string"
      }
    }
  ]
}
```

**Implementation options:**
- `mcp` Python package with HTTP transport
- Custom FastAPI/Django endpoint mimicking MCP protocol
- Adapter that bridges Django to MCP

### 5. Audio Storage

- Files stored on disk (or S3 for production)
- Django `FileField` handles upload
- Track model stores path + metadata
- Consider cleanup job for old/orphaned files

## Future Extensions

- **VAMPNET integration** - audio manipulation/generation
- **Comparison tools** - analyse differences between tracks
- **Visualisation endpoints** - return spectrograms, waveforms as images
- **Streaming analysis** - for long files
- **User sessions** - track ownership, history

## Tech Stack

- **Django + DRF** - API
- **PostgreSQL** - database (SQLite for dev)
- **librosa, numpy, soundfile** - audio analysis
- **MCP Python SDK** - tool exposure
- **LLM client** - Anthropic/OpenAI/Google SDK (orchestrator uses this)

## Project Structure

```
audio-analysis-agents/
├── manage.py
├── pyproject.toml
├── config/                 # Django settings
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── tracks/                 # Django app for audio management
│   ├── models.py
│   ├── views.py
│   ├── serializers.py
│   └── urls.py
├── analysis/               # Django app for analysis
│   ├── views.py
│   └── urls.py
├── agents/                 # Standalone analysis agents (no Django dep)
│   ├── base.py
│   ├── spectral.py
│   ├── temporal.py
│   └── rhythm.py
├── orchestrator/           # Routing logic
│   ├── orchestrator.py
│   └── routing.py
├── mcp_server/             # MCP HTTP server
│   └── server.py
├── tests/
└── docs/
    └── DESIGN.md
```

## Open Questions

1. **LLM provider** - which to use for orchestrator routing? (Claude, Gemini, local?)
2. **Caching strategy** - cache all analysis on upload, or on-demand?
3. **MCP auth** - how to secure the MCP endpoint?
4. **Frontend tech** - what's the existing frontend built with?
