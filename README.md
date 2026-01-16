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

**Core:**
- `librosa` - Audio analysis
- `numpy` - Numerical operations
- `soundfile` - Audio file I/O
- `pydantic` - Data validation

**API (`pip install -e ".[api]"`):**
- `django` - Web framework
- `djangorestframework` - REST API
- `djangorestframework-simplejwt` - JWT authentication
- `channels` - WebSocket support
- `gunicorn` / `uvicorn` - Production server
- `whitenoise` - Static file serving
- `dj-database-url` - Database URL parsing
- `psycopg2-binary` - PostgreSQL driver
- `django-storages` - S3 storage backend
- `boto3` - AWS SDK

## Authentication

The API supports two authentication methods:

### JWT Authentication (Recommended)

**Demo Mode** - Get a temporary token with limited usage (15 min, 5 requests):
```bash
curl -X POST https://your-app.railway.app/api/auth/demo/ \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Login** - Get a token for registered users:
```bash
curl -X POST https://your-app.railway.app/api/auth/login/ \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "secret"}'
```

**Using the token:**
```bash
curl https://your-app.railway.app/api/tracks/list/ \
  -H "Authorization: Bearer <access_token>"
```

**Refresh token:**
```bash
curl -X POST https://your-app.railway.app/api/auth/refresh/ \
  -H "Content-Type: application/json" \
  -d '{"refresh": "<refresh_token>"}'
```

### API Key Authentication (Legacy)

```bash
curl https://your-app.railway.app/api/tracks/list/ \
  -H "X-API-Key: your-api-key"
```

## Railway Deployment

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DJANGO_SECRET_KEY` | Yes | Generate a secure random key (50+ chars) |
| `DJANGO_DEBUG` | Yes | Set to `false` for production |
| `DJANGO_ALLOWED_HOSTS` | Yes | Set to `.railway.app` (or your custom domain) |
| `DATABASE_URL` | Auto | Provided automatically by Railway PostgreSQL |
| `API_KEY` | No | API key for legacy authentication |
| `LLM_MODEL` | No | LLM model to use (default: `gemini/gemini-2.0-flash`) |
| `GOOGLE_API_KEY` | Conditional | Required if using Gemini models |
| `ANTHROPIC_API_KEY` | Conditional | Required if using Claude models |
| `OPIK_API_KEY` | No | Opik API key for tracing (omit to disable) |
| `OPIK_PROJECT_NAME` | No | Opik project name (default: `audio-analysis-agents`) |

**S3 Storage (recommended for Railway):**

| Variable | Required | Description |
|----------|----------|-------------|
| `AWS_STORAGE_BUCKET_NAME` | Yes* | S3 bucket name (enables S3 storage) |
| `AWS_ACCESS_KEY_ID` | Yes* | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | Yes* | AWS secret key |
| `AWS_S3_REGION_NAME` | No | AWS region (default: `us-east-1`) |

*Required if using S3 storage. If not set, falls back to local filesystem.

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

#### 3. Set Up S3 Storage (Recommended)

Use Terraform to create an S3 bucket with the right permissions:

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your bucket name

terraform init
terraform apply

# Get credentials for Railway
terraform output -raw aws_secret_access_key
```

Then add the S3 variables to Railway (see Environment Variables above).

**Alternative: Railway Volume** (simpler but less reliable)

- Click on your app service → **Settings** tab
- Scroll to **Volumes** section → **+ Add Volume**
- Set mount path: `/app/data/uploads`
- Add variable: `MEDIA_ROOT` = `/app/data/uploads`

#### 4. Set Environment Variables

In your app service → **Variables** tab, add:

```
DJANGO_SECRET_KEY=<generate with: python -c "import secrets; print(secrets.token_urlsafe(50))">
DJANGO_DEBUG=false
DJANGO_ALLOWED_HOSTS=.railway.app
GOOGLE_API_KEY=<your-google-api-key>

# S3 Storage (from terraform output)
AWS_STORAGE_BUCKET_NAME=<your-bucket-name>
AWS_ACCESS_KEY_ID=<from-terraform>
AWS_SECRET_ACCESS_KEY=<from-terraform>
AWS_S3_REGION_NAME=us-east-1
```

## License

MIT
