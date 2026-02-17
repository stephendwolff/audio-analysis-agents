# HTTP API

The Sketchbook API serves audio analysis over HTTP. Clients submit audio fragments, the server analyses them asynchronously, and clients poll for results.

## Authentication

JWT via djangorestframework-simplejwt. The health endpoint is public; all other endpoints require a valid token.

```
POST /api/auth/login/   ->  { "access": "<jwt>", "refresh": "<jwt>" }

Authorization: Bearer <jwt-access-token>
```

Demo tokens (15 min, 5 requests) are available for testing:

```
POST /api/auth/demo/    ->  { "access": "<jwt>", "refresh": "<jwt>" }
```

## Endpoints

### GET /api/health/

Server availability check. No authentication required.

**Response: 200 OK**

```json
{
    "status": "ok",
    "version": "1.0.0",
    "agents": ["spectral", "temporal", "rhythm"]
}
```

**Response: 503 Service Unavailable** -- if agents fail to load.

---

### POST /api/analyse/

Submit an audio fragment for analysis. Returns immediately; analysis runs in background via Celery.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| audio | binary | yes | Audio file bytes |
| mime_type | string | yes | e.g. `audio/wav`, `audio/m4a`, `audio/webm` |
| fragment_id | string | yes | Client-generated UUID |
| duration_seconds | number | yes | Duration of the audio |

**Supported formats:** wav, mp3, flac, ogg, mp4, m4a, webm, aac

**Response: 202 Accepted** -- fragment queued for analysis

```json
{
    "fragment_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "pending",
    "poll_url": "/api/analyse/550e8400-e29b-41d4-a716-446655440000/"
}
```

**Response: 200 OK** -- fragment already analysed (idempotent)

Returns the full result, same shape as the GET complete response below.

**Response: 400 Bad Request** -- missing or invalid fields

```json
{
    "error": "validation_error",
    "message": "audio file is required"
}
```

**Response: 422 Unprocessable Entity** -- unsupported audio format

```json
{
    "error": "unsupported_format",
    "message": "Cannot decode audio/ogg"
}
```

---

### GET /api/analyse/{fragment_id}/

Poll for analysis results.

**Response: 200 OK** -- analysis complete

```json
{
    "fragment_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "complete",
    "model_version": "1.0.0",
    "dimensions": {
        "bpm": 92.4,
        "time_signature": "4/4",
        "swing": 0.7,
        "steadiness": 0.6,
        "upbeat": false
    },
    "descriptors": ["swung", "laid-back", "steady"],
    "raw_data": {
        "beats": [0.52, 1.17, 1.82, 2.48],
        "onsets": [0.01, 0.52, 0.78, 1.17],
        "spectral_centroid_mean": 2400.5,
        "rms_energy_mean": 0.034
    }
}
```

**Response: 200 OK** -- still processing

```json
{
    "fragment_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "pending"
}
```

**Response: 200 OK** -- analysis failed

```json
{
    "fragment_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "failed",
    "error": "Analysis timed out"
}
```

**Response: 404 Not Found** -- unknown fragment ID.

## Response Shape

The `dimensions` keys come from the rhythm agent's output and may evolve. Clients should treat them as a flexible key-value bag.

`descriptors` are human-readable tags generated from dimension values using rule-based thresholds (e.g. BPM > 140 = "driving", swing > 0.4 = "swung").

`raw_data` contains beat positions, onset positions, and summary statistics from the spectral and temporal agents.

## Idempotency

The `fragment_id` (client-generated UUID) provides idempotency:

- **First POST** with a fragment_id: creates fragment, queues analysis, returns 202
- **POST while pending/analysing**: returns 202 with poll URL (no duplicate work)
- **POST after complete**: returns 200 with the cached result
- **POST after failed**: re-queues analysis, returns 202

## Polling Strategy

After submitting, poll `GET /api/analyse/{fragment_id}/` at 2-second intervals for up to 30 seconds. If still pending, back off and retry on next app launch.

## Background Processing

Analysis runs as a Celery task (`analyse_fragment`). The task:

1. Sets fragment status to `analyzing`
2. Loads audio, converts to wav if needed
3. Runs rhythm, spectral, and temporal agents
4. Builds the response (dimensions, descriptors, raw_data)
5. Sets status to `complete` (or `failed` on error)

Requires a running Celery worker and Redis broker.
