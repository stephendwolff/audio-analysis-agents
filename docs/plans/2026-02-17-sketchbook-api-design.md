# Musical Sketchbook Analysis API — Design

Adapt the existing audio-analysis-agents codebase to serve the Musical Sketchbook iOS app's analysis contract (v0.2.0).

---

## API Contract Summary

Three endpoints, all under `/api/`:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/analyse` | POST | Submit a fragment for background analysis |
| `/api/analyse/{fragment_id}` | GET | Poll for results |
| `/api/health` | GET | Server availability check |

**Auth:** JWT via the existing `djangorestframework-simplejwt` setup.

**Async flow:** POST returns 202 immediately. The iOS app polls GET until the status changes to `complete` or `failed`. Re-POSTing an already-complete fragment returns 200 with the cached result.

Full contract: `docs/api-contract.md` in the Musical Sketchbook repo.

---

## Architecture

### New Django App: `src/sketchbook/`

```
src/sketchbook/
├── __init__.py
├── apps.py
├── models.py          # Fragment model
├── views.py           # POST /analyse, GET /analyse/{id}, GET /health
├── urls.py
├── tasks.py           # analyse_fragment Celery task
├── serializers.py     # Maps agent output to contract response shape
└── descriptors.py     # Rule-based descriptor generation
```

The sketchbook app remains independent from `src/api/`. It reuses the analysis agents and audio utilities but does not depend on the Track model, WebSocket consumer, or LLM orchestrator.

### Modified Existing Files

- `src/agents/rhythm.py` — Add time_signature, swing, steadiness, upbeat; remove array truncation
- `src/tools/audio_utils.py` — Add mp4/webm decoding via pydub (ffmpeg)
- `config/urls.py` — Include sketchbook URLs
- `config/settings.py` — Register sketchbook app
- `pyproject.toml` — Add pydub dependency

---

## Fragment Model

```python
class Fragment(models.Model):
    fragment_id = models.UUIDField(unique=True, db_index=True)  # iOS app's UUID
    user = models.ForeignKey(User, on_delete=models.CASCADE)

    audio_storage_path = models.CharField(max_length=500)
    mime_type = models.CharField(max_length=100)
    duration_seconds = models.FloatField()

    class Status(models.TextChoices):
        PENDING = "pending"
        ANALYZING = "analyzing"
        COMPLETE = "complete"
        FAILED = "failed"

    status = models.CharField(max_length=20, choices=Status, default=Status.PENDING)
    error_message = models.TextField(blank=True)

    model_version = models.CharField(max_length=20)
    analysis = models.JSONField(default=dict)

    created_at = models.DateTimeField(auto_now_add=True)
```

`fragment_id` uniquely identifies the iOS app's audio fragment. The unique constraint enforces idempotency: a second POST with the same fragment_id retrieves the existing record rather than creating a duplicate.

---

## Request Flow

### POST /api/analyse

1. Validate required fields (audio, mime_type, fragment_id, duration_seconds)
2. Check mime_type is supported; return 422 if not
3. Look up existing Fragment by fragment_id:
   - **COMPLETE** — return 200 with cached result
   - **PENDING or ANALYZING** — return 202 with poll_url
   - **FAILED** — reset to PENDING and re-queue for retry
   - **Not found** — continue
4. Write the audio file to Django storage
5. Create a Fragment record with status=PENDING
6. Queue `analyse_fragment.delay(fragment_id)`, or return 503 if Celery is unreachable
7. Return 202 with `{"fragment_id": "...", "status": "pending", "poll_url": "/api/analyse/{id}"}`

### GET /api/analyse/{fragment_id}

1. Look up the Fragment; return 404 if not found
2. Return the response based on status:
   - **PENDING / ANALYZING** — `{"fragment_id": "...", "status": "pending"}`
   - **COMPLETE** — full result with dimensions, descriptors, and raw_data
   - **FAILED** — `{"fragment_id": "...", "status": "failed", "error": "..."}`

### GET /api/health

Return `{"status": "ok", "version": "1.0.0", "agents": ["rhythm", "spectral", "temporal"]}`. Verify that agents instantiate successfully from the registry. Return 503 if the registry is empty or any agent fails to instantiate.

---

## Enhanced RhythmAgent

The existing agent returns tempo_bpm, beat_times (truncated to 20), onset_times (truncated to 30), and tempo_stability. Add four new fields:

### time_signature

Detect beat grouping by analyzing onset strength patterns relative to beat positions. Count strong onsets per beat cycle to distinguish 3/4 from 4/4. Use `librosa.beat.beat_track` positions and onset strength envelope. Return a string: `"4/4"`, `"3/4"`, or `"other"`.

### swing

Compare actual beat subdivisions to an even grid. For each pair of consecutive eighth-note positions, measure the ratio of long-to-short intervals. A perfectly straight rhythm has ratio 1:1 (swing=0.0); full triplet swing has ratio 2:1 (swing=1.0). Compute from inter-onset intervals between beats.

### steadiness

The agent already computes `tempo_stability.cv` (coefficient of variation of inter-beat intervals). Steadiness inverts this value and normalizes it to 0–1:

```python
steadiness = max(0.0, 1.0 - (cv / 0.5))  # cv of 0 → 1.0, cv of 0.5+ → 0.0
```

### upbeat

Detect anacrusis by checking whether significant onsets occur before the first detected beat. Compare the first onset time to the first beat time. If onsets precede the first beat by more than half a beat interval, set `upbeat = True`.

### Array truncation

Remove the `[:20]` and `[:30]` limits on beat_times and onset_times. For 60-second audio at 120 BPM, this yields roughly 120 beats and 200 onsets, which remain small enough for JSON transmission.

---

## Celery Task: analyse_fragment

```python
@shared_task(bind=True, max_retries=3, default_retry_delay=30)
def analyse_fragment(self, fragment_id_str):
    fragment = Fragment.objects.get(fragment_id=fragment_id_str)
    fragment.status = Fragment.Status.ANALYZING
    fragment.save(update_fields=["status"])

    # Load audio (extended load_audio handles mp4/webm via pydub)
    audio = load_audio(fragment.audio_storage_path)

    # Run agents
    rhythm = RhythmAgent().analyse(audio.samples, audio.sample_rate)
    spectral = SpectralAgent().analyse(audio.samples, audio.sample_rate)
    temporal = TemporalAgent().analyse(audio.samples, audio.sample_rate)

    # Build contract response and store
    fragment.analysis = build_analysis_result(rhythm, spectral, temporal)
    fragment.status = Fragment.Status.COMPLETE
    fragment.save(update_fields=["analysis", "status"])
```

On exception: set status to FAILED with an error message, then raise to trigger Celery retry.

---

## Response Builder

Transform agent output into the contract response shape:

```python
def build_analysis_result(rhythm, spectral, temporal):
    dimensions = {
        "bpm": rhythm.data["tempo_bpm"],
        "time_signature": rhythm.data["time_signature"],
        "swing": rhythm.data["swing"],
        "steadiness": rhythm.data["steadiness"],
        "upbeat": rhythm.data["upbeat"],
    }
    return {
        "dimensions": dimensions,
        "descriptors": generate_descriptors(dimensions),
        "raw_data": {
            "beats": rhythm.data["beat_times"],
            "onsets": rhythm.data["onset_times"],
            "spectral_centroid_mean": spectral.data["spectral_centroid"]["mean"],
            "rms_energy_mean": temporal.data["rms_energy"]["mean"],
        },
    }
```

---

## Descriptors

Generate descriptors using rule-based logic applied to dimension values:

| Condition | Descriptor |
|-----------|------------|
| bpm > 140 | "driving" |
| bpm < 90 | "laid-back" |
| 90 <= bpm <= 140 | "moderate-tempo" |
| swing > 0.4 | "swung" |
| swing < 0.1 | "straight" |
| steadiness > 0.8 | "steady" |
| steadiness < 0.4 | "loose" |
| upbeat is True | "upbeat-start" |

Multiple descriptors apply when conditions match. For example, a fragment at 85 BPM with swing 0.6 and steadiness 0.7 produces: `["laid-back", "swung"]`.

---

## Audio Decoding Extension

Extend `load_audio()` in `src/tools/audio_utils.py` to handle audio container formats that libsndfile cannot decode directly:

```python
NEEDS_CONVERSION = {"audio/mp4", "audio/m4a", "audio/webm", "audio/aac"}

if mime_type in NEEDS_CONVERSION:
    from pydub import AudioSegment
    segment = AudioSegment.from_file(file_obj, format=format_from_mime(mime_type))
    wav_buffer = io.BytesIO()
    segment.export(wav_buffer, format="wav")
    wav_buffer.seek(0)
    # Proceed with existing librosa/soundfile loading from wav_buffer
```

Requires `pydub` (Python) and `ffmpeg` (system). Document the ffmpeg requirement in the README.

**Supported formats:** audio/wav, audio/mp3, audio/flac, audio/ogg, audio/mp4, audio/m4a, audio/webm, and audio/aac.

---

## Error Handling

| Scenario | Status | Response |
|----------|--------|----------|
| Missing required field | 400 | `{"error": "validation_error", "message": "..."}` |
| Unsupported mime_type | 422 | `{"error": "unsupported_format", "message": "Cannot decode audio/xyz"}` |
| Invalid or missing JWT | 401 | Standard DRF response |
| Unknown fragment_id (GET) | 404 | `{"detail": "Not found."}` |
| Celery unreachable (POST) | 503 | `{"error": "service_unavailable", "message": "..."}` |
| Agent crash during analysis | — | Fragment.status=FAILED, error stored, Celery retries up to 3 times |

---

## Configuration

```python
# config/settings.py or src/sketchbook/apps.py
SKETCHBOOK_MODEL_VERSION = "1.0.0"  # Bump when analysis logic changes
```

---

## Decisions Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| App structure | Separate `src/sketchbook/` app | Clean separation from existing chat/upload API |
| Analysis gaps | Enhance existing RhythmAgent | Keeps rhythm logic in one place |
| Data model | New Fragment model | Fragments are conceptually different from Tracks |
| Descriptors | Rule-based | Predictable, no LLM dependency, low latency |
| Audio decoding | Extend load_audio() with pydub | Centralised audio handling |
| Auth | Existing JWT (simplejwt) | Already set up, iOS app calls /api/auth/login/ |
| Model version | Semver from config | Simple, manually bumped |
| URL prefix | /api/ (matching existing API) | Consistency with existing endpoints |
