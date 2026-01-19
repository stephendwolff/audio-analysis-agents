# Router Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor LLM routing to use generated tool schemas, strategy pattern for analysis, and async streaming.

**Architecture:** Agents expose metadata for tool generation. AnalysisProvider ABC abstracts data source (cached/live). Async routers yield events for real-time WebSocket streaming.

**Tech Stack:** Python 3.11+, Django, LiteLLM, pytest, pytest-asyncio

---

### Task 1: Add Agent Description to Base Class

**Files:**
- Modify: `src/agents/base.py`
- Test: `tests/test_agents/test_base.py`

**Step 1: Write the failing test**

```python
# tests/test_agents/test_base.py
import pytest
from src.agents.base import BaseAgent, AnalysisResult


def test_base_agent_requires_description():
    """BaseAgent subclasses must define description."""

    class IncompleteAgent(BaseAgent):
        name = "incomplete"
        # Missing description

        def analyse(self, samples, sample_rate):
            return AnalysisResult(success=True, data={})

    with pytest.raises(TypeError, match="description"):
        IncompleteAgent()


def test_base_agent_with_description_instantiates():
    """BaseAgent subclasses with description can be instantiated."""

    class CompleteAgent(BaseAgent):
        name = "complete"
        description = "A complete test agent"

        def analyse(self, samples, sample_rate):
            return AnalysisResult(success=True, data={})

    agent = CompleteAgent()
    assert agent.name == "complete"
    assert agent.description == "A complete test agent"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_agents/test_base.py -v`
Expected: FAIL - BaseAgent doesn't enforce description

**Step 3: Read the existing base.py**

Read `src/agents/base.py` to understand current structure.

**Step 4: Write minimal implementation**

Update `src/agents/base.py` to add abstract `description` class variable:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class AnalysisResult:
    """Result from an analysis agent."""
    success: bool
    data: dict[str, Any] | None = None
    error: str | None = None


class BaseAgent(ABC):
    """Base class for audio analysis agents."""

    name: str
    description: str  # Add this line

    def __init__(self):
        # Validate required class attributes
        if not hasattr(self, 'name') or not self.name:
            raise TypeError(f"{self.__class__.__name__} must define 'name'")
        if not hasattr(self, 'description') or not self.description:
            raise TypeError(f"{self.__class__.__name__} must define 'description'")

    @abstractmethod
    def analyse(self, samples: np.ndarray, sample_rate: int) -> AnalysisResult:
        """Analyse audio samples."""
        ...
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_agents/test_base.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/agents/base.py tests/test_agents/test_base.py
git commit -m "feat(agents): add required description field to BaseAgent"
```

---

### Task 2: Add Descriptions to Existing Agents

**Files:**
- Modify: `src/agents/spectral.py`
- Modify: `src/agents/temporal.py`
- Modify: `src/agents/rhythm.py`
- Test: `tests/test_agents/test_agent_descriptions.py`

**Step 1: Write the failing test**

```python
# tests/test_agents/test_agent_descriptions.py
import pytest
from src.agents import SpectralAgent, TemporalAgent, RhythmAgent


@pytest.mark.parametrize("agent_class,expected_name", [
    (SpectralAgent, "spectral"),
    (TemporalAgent, "temporal"),
    (RhythmAgent, "rhythm"),
])
def test_agent_has_name_and_description(agent_class, expected_name):
    """All agents must have name and description."""
    agent = agent_class()
    assert agent.name == expected_name
    assert isinstance(agent.description, str)
    assert len(agent.description) > 20  # Meaningful description
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_agents/test_agent_descriptions.py -v`
Expected: FAIL - agents don't have description

**Step 3: Read existing agent files**

Read `src/agents/spectral.py`, `src/agents/temporal.py`, `src/agents/rhythm.py` to see current structure.

**Step 4: Add descriptions to each agent**

Update `src/agents/spectral.py`:
```python
class SpectralAgent(BaseAgent):
    """Agent for spectral/frequency analysis."""

    name = "spectral"
    description = (
        "Analyse frequency content of audio. Returns spectral centroid (brightness), "
        "bandwidth, rolloff, flatness (tonal vs noisy), dominant frequencies, and MFCC summary. "
        "Use for questions about frequency, pitch, tone, timbre, or 'what does it sound like'."
    )
```

Update `src/agents/temporal.py`:
```python
class TemporalAgent(BaseAgent):
    """Agent for time-domain analysis."""

    name = "temporal"
    description = (
        "Analyse time-domain properties of audio. Returns duration, RMS energy (loudness), "
        "peak amplitude, dynamic range, zero crossing rate, and amplitude envelope. "
        "Use for questions about duration, volume, dynamics, loudness, or energy."
    )
```

Update `src/agents/rhythm.py`:
```python
class RhythmAgent(BaseAgent):
    """Agent for rhythm and tempo analysis."""

    name = "rhythm"
    description = (
        "Analyse tempo and rhythmic properties of audio. Returns estimated BPM, "
        "beat positions, onset times, and tempo stability. "
        "Use for questions about tempo, BPM, beats, rhythm, or timing."
    )
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_agents/test_agent_descriptions.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/agents/spectral.py src/agents/temporal.py src/agents/rhythm.py tests/test_agents/test_agent_descriptions.py
git commit -m "feat(agents): add descriptions to spectral, temporal, rhythm agents"
```

---

### Task 3: Add Tool Schema Generation to Registry

**Files:**
- Modify: `src/agents/registry.py`
- Test: `tests/test_agents/test_registry.py`

**Step 1: Write the failing test**

```python
# tests/test_agents/test_registry.py
import pytest
from src.agents.registry import (
    AGENT_REGISTRY,
    register_agent,
    get_all_agents,
    get_agent,
    get_tool_schemas_openai,
)
from src.agents.base import BaseAgent, AnalysisResult


class TestToolSchemaGeneration:
    """Tests for generating tool schemas from registry."""

    def test_get_tool_schemas_openai_returns_list(self):
        """Should return a list of tool schemas."""
        schemas = get_tool_schemas_openai()
        assert isinstance(schemas, list)

    def test_get_tool_schemas_openai_schema_structure(self):
        """Each schema should have correct OpenAI format."""
        schemas = get_tool_schemas_openai()

        # Should have at least the 3 registered agents
        assert len(schemas) >= 3

        for schema in schemas:
            assert schema["type"] == "function"
            assert "function" in schema
            func = schema["function"]
            assert "name" in func
            assert func["name"].startswith("analyse_")
            assert "description" in func
            assert len(func["description"]) > 20
            assert "parameters" in func
            params = func["parameters"]
            assert params["type"] == "object"
            assert "track_id" in params["properties"]
            assert "track_id" in params["required"]

    def test_get_tool_schemas_openai_includes_all_agents(self):
        """Should include schemas for all registered agents."""
        schemas = get_tool_schemas_openai()
        schema_names = {s["function"]["name"] for s in schemas}

        assert "analyse_spectral" in schema_names
        assert "analyse_temporal" in schema_names
        assert "analyse_rhythm" in schema_names
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_agents/test_registry.py::TestToolSchemaGeneration -v`
Expected: FAIL - get_tool_schemas_openai doesn't exist

**Step 3: Read existing registry.py**

Read `src/agents/registry.py` to see current structure.

**Step 4: Add tool schema generation**

Update `src/agents/registry.py`:

```python
"""Agent registry with tool schema generation."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseAgent

AGENT_REGISTRY: dict[str, type["BaseAgent"]] = {}


def register_agent(cls):
    """Decorator to register an agent for automatic analysis."""
    AGENT_REGISTRY[cls.name] = cls
    return cls


def get_all_agents() -> list["BaseAgent"]:
    """Return instances of all registered agents."""
    return [cls() for cls in AGENT_REGISTRY.values()]


def get_agent(name: str) -> "BaseAgent | None":
    """Get a specific agent by name."""
    cls = AGENT_REGISTRY.get(name)
    return cls() if cls else None


def get_tool_schemas_openai() -> list[dict]:
    """
    Generate OpenAI-format tool schemas from registered agents.

    Returns a list of tool definitions compatible with OpenAI/LiteLLM.
    """
    schemas = []
    for cls in AGENT_REGISTRY.values():
        agent = cls()
        schemas.append({
            "type": "function",
            "function": {
                "name": f"analyse_{agent.name}",
                "description": agent.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "track_id": {
                            "type": "string",
                            "description": "Track ID to analyse"
                        }
                    },
                    "required": ["track_id"]
                }
            }
        })
    return schemas
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_agents/test_registry.py::TestToolSchemaGeneration -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/agents/registry.py tests/test_agents/test_registry.py
git commit -m "feat(agents): add OpenAI tool schema generation to registry"
```

---

### Task 4: Add Fat Model Methods to Track

**Files:**
- Modify: `src/api/models.py`
- Test: `tests/test_api/test_models.py`

**Step 1: Write the failing test**

```python
# tests/test_api/test_models.py
import pytest
from django.test import TestCase
from src.api.models import Track


class TestTrackModel(TestCase):
    """Tests for Track model methods."""

    def test_get_or_none_returns_track(self):
        """get_or_none returns track when it exists."""
        track = Track.objects.create(
            original_filename="test.wav",
            storage_path="test.wav",
            file_size=1000,
        )
        result = Track.get_or_none(str(track.id))
        assert result == track

    def test_get_or_none_returns_none_for_missing(self):
        """get_or_none returns None when track doesn't exist."""
        result = Track.get_or_none("00000000-0000-0000-0000-000000000000")
        assert result is None

    def test_is_ready_true_when_ready(self):
        """is_ready returns True when status is READY."""
        track = Track.objects.create(
            original_filename="test.wav",
            storage_path="test.wav",
            file_size=1000,
            status=Track.Status.READY,
        )
        assert track.is_ready is True

    def test_is_ready_false_when_pending(self):
        """is_ready returns False when status is not READY."""
        track = Track.objects.create(
            original_filename="test.wav",
            storage_path="test.wav",
            file_size=1000,
            status=Track.Status.PENDING,
        )
        assert track.is_ready is False

    def test_has_analysis_true_when_present(self):
        """has_analysis returns True when analysis exists."""
        track = Track.objects.create(
            original_filename="test.wav",
            storage_path="test.wav",
            file_size=1000,
            analysis={"rhythm": {"bpm": 120}},
        )
        assert track.has_analysis("rhythm") is True

    def test_has_analysis_false_when_missing(self):
        """has_analysis returns False when analysis doesn't exist."""
        track = Track.objects.create(
            original_filename="test.wav",
            storage_path="test.wav",
            file_size=1000,
            analysis={},
        )
        assert track.has_analysis("rhythm") is False

    def test_has_analysis_false_when_error(self):
        """has_analysis returns False when analysis contains error."""
        track = Track.objects.create(
            original_filename="test.wav",
            storage_path="test.wav",
            file_size=1000,
            analysis={"rhythm": {"error": "Failed"}},
        )
        assert track.has_analysis("rhythm") is False

    def test_get_analysis_returns_data(self):
        """get_analysis returns analysis data."""
        track = Track.objects.create(
            original_filename="test.wav",
            storage_path="test.wav",
            file_size=1000,
            analysis={"rhythm": {"bpm": 120}},
        )
        result = track.get_analysis("rhythm")
        assert result == {"bpm": 120}

    def test_get_analysis_raises_key_error_when_missing(self):
        """get_analysis raises KeyError when analysis doesn't exist."""
        track = Track.objects.create(
            original_filename="test.wav",
            storage_path="test.wav",
            file_size=1000,
            analysis={},
        )
        with pytest.raises(KeyError):
            track.get_analysis("rhythm")

    def test_get_analysis_raises_value_error_on_error(self):
        """get_analysis raises ValueError when analysis has error."""
        track = Track.objects.create(
            original_filename="test.wav",
            storage_path="test.wav",
            file_size=1000,
            analysis={"rhythm": {"error": "Analysis failed"}},
        )
        with pytest.raises(ValueError, match="Analysis failed"):
            track.get_analysis("rhythm")

    def test_set_analysis_stores_data(self):
        """set_analysis stores analysis and saves to DB."""
        track = Track.objects.create(
            original_filename="test.wav",
            storage_path="test.wav",
            file_size=1000,
        )
        track.set_analysis("rhythm", {"bpm": 120})

        # Refresh from DB
        track.refresh_from_db()
        assert track.analysis["rhythm"] == {"bpm": 120}
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_api/test_models.py::TestTrackModel -v`
Expected: FAIL - methods don't exist

**Step 3: Read existing models.py**

Read `src/api/models.py` to see current Track model.

**Step 4: Add methods to Track model**

Update `src/api/models.py`:

```python
"""
API Models
"""

import uuid

from django.contrib.auth.models import User
from django.db import models


class Track(models.Model):
    """Audio track with cached analysis results."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4)

    # File info
    original_filename = models.CharField(max_length=255)
    storage_path = models.CharField(max_length=500)
    file_url = models.URLField(max_length=500, blank=True)
    file_size = models.PositiveIntegerField()
    duration = models.FloatField(null=True)

    # Ownership
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)

    # Status
    class Status(models.TextChoices):
        PENDING = "pending"
        ANALYZING = "analyzing"
        READY = "ready"
        FAILED = "failed"

    status = models.CharField(max_length=20, choices=Status.choices, default=Status.PENDING)
    status_message = models.TextField(blank=True)

    # Analysis results
    analysis = models.JSONField(default=dict)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    analyzed_at = models.DateTimeField(null=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.original_filename} ({self.status})"

    # ----- Class methods -----

    @classmethod
    def get_or_none(cls, track_id: str) -> "Track | None":
        """Get track by ID, returning None if not found."""
        try:
            return cls.objects.get(id=track_id)
        except cls.DoesNotExist:
            return None

    # ----- Properties -----

    @property
    def file_path(self) -> str:
        """Resolve storage path to filesystem path or URL."""
        from django.core.files.storage import default_storage
        if hasattr(default_storage, "path"):
            try:
                return default_storage.path(self.storage_path)
            except NotImplementedError:
                pass
        return self.file_url

    @property
    def is_ready(self) -> bool:
        """Check if track is ready for questions."""
        return self.status == self.Status.READY

    # ----- Analysis methods -----

    def has_analysis(self, agent_name: str) -> bool:
        """Check if analysis exists and has no error."""
        if agent_name not in self.analysis:
            return False
        data = self.analysis[agent_name]
        return not (isinstance(data, dict) and "error" in data)

    def get_analysis(self, agent_name: str) -> dict:
        """Get analysis result. Raises KeyError/ValueError if unavailable."""
        if agent_name not in self.analysis:
            raise KeyError(f"No {agent_name} analysis for track {self.id}")
        data = self.analysis[agent_name]
        if isinstance(data, dict) and "error" in data:
            raise ValueError(data["error"])
        return data

    def set_analysis(self, agent_name: str, data: dict) -> None:
        """Store analysis result."""
        self.analysis[agent_name] = data
        self.save(update_fields=["analysis"])

    def queue_analysis(self) -> str | None:
        """Queue background analysis task. Returns task_id or None on failure."""
        from src.tasks.analysis import analyze_track
        try:
            result = analyze_track.delay(str(self.id))
            return result.id
        except Exception as e:
            self.status = self.Status.FAILED
            self.status_message = f"Failed to queue analysis: {e}"
            self.save(update_fields=["status", "status_message"])
            return None


class UserProfile(models.Model):
    """
    Extended user profile for tracking demo usage.
    """

    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="profile")
    is_demo = models.BooleanField(default=False)
    request_count = models.IntegerField(default=0)
    request_limit = models.IntegerField(default=5)

    def __str__(self):
        return f"{self.user.username} ({'demo' if self.is_demo else 'user'})"

    def can_make_request(self) -> bool:
        """Check if user can make another request."""
        if not self.is_demo:
            return True
        return self.request_count < self.request_limit

    def increment_request_count(self) -> None:
        """Increment request count for demo users."""
        if self.is_demo:
            self.request_count += 1
            self.save(update_fields=["request_count"])
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_api/test_models.py::TestTrackModel -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/api/models.py tests/test_api/test_models.py
git commit -m "feat(models): add fat model methods to Track"
```

---

### Task 5: Create Router Events

**Files:**
- Create: `src/orchestrator/events.py`
- Test: `tests/test_orchestrator/test_events.py`

**Step 1: Write the failing test**

```python
# tests/test_orchestrator/test_events.py
import pytest
from src.orchestrator.events import (
    ToolCallEvent,
    ToolResultEvent,
    TokenEvent,
    DoneEvent,
    ErrorEvent,
    RouterEvent,
)


def test_tool_call_event_structure():
    """ToolCallEvent has correct structure."""
    event: ToolCallEvent = {"type": "tool_call", "tool": "analyse_rhythm"}
    assert event["type"] == "tool_call"
    assert event["tool"] == "analyse_rhythm"


def test_tool_result_event_structure():
    """ToolResultEvent has correct structure."""
    event: ToolResultEvent = {"type": "tool_result", "tool": "analyse_rhythm", "success": True}
    assert event["type"] == "tool_result"
    assert event["success"] is True


def test_token_event_structure():
    """TokenEvent has correct structure."""
    event: TokenEvent = {"type": "token", "text": "The"}
    assert event["type"] == "token"
    assert event["text"] == "The"


def test_done_event_structure():
    """DoneEvent has correct structure."""
    event: DoneEvent = {"type": "done", "full_response": "The tempo is 120 BPM"}
    assert event["type"] == "done"
    assert event["full_response"] == "The tempo is 120 BPM"


def test_error_event_structure():
    """ErrorEvent has correct structure."""
    event: ErrorEvent = {"type": "error", "message": "Something went wrong"}
    assert event["type"] == "error"
    assert event["message"] == "Something went wrong"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_orchestrator/test_events.py -v`
Expected: FAIL - module doesn't exist

**Step 3: Create events module**

Create `src/orchestrator/events.py`:

```python
"""
Router event types for streaming responses.

Events are yielded by routers during question processing,
allowing real-time updates to WebSocket clients.
"""

from typing import TypedDict, Literal


class ToolCallEvent(TypedDict):
    """Emitted when the LLM requests a tool call."""
    type: Literal["tool_call"]
    tool: str


class ToolResultEvent(TypedDict):
    """Emitted after a tool has been executed."""
    type: Literal["tool_result"]
    tool: str
    success: bool


class TokenEvent(TypedDict):
    """Emitted for each token in the streaming response."""
    type: Literal["token"]
    text: str


class DoneEvent(TypedDict):
    """Emitted when the response is complete."""
    type: Literal["done"]
    full_response: str


class ErrorEvent(TypedDict):
    """Emitted when an error occurs."""
    type: Literal["error"]
    message: str


# Union type for all router events
RouterEvent = ToolCallEvent | ToolResultEvent | TokenEvent | DoneEvent | ErrorEvent
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_orchestrator/test_events.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/orchestrator/events.py tests/test_orchestrator/test_events.py
git commit -m "feat(orchestrator): add router event types"
```

---

### Task 6: Create AnalysisProvider ABC and CachedProvider

**Files:**
- Create: `src/orchestrator/providers.py`
- Test: `tests/test_orchestrator/test_providers.py`

**Step 1: Write the failing test**

```python
# tests/test_orchestrator/test_providers.py
import pytest
from django.test import TestCase
from src.orchestrator.providers import (
    AnalysisProvider,
    CachedAnalysisProvider,
)
from src.api.models import Track


class TestCachedAnalysisProvider(TestCase):
    """Tests for CachedAnalysisProvider."""

    def test_get_analysis_returns_cached_data(self):
        """get_analysis returns data from track.analysis."""
        track = Track.objects.create(
            original_filename="test.wav",
            storage_path="test.wav",
            file_size=1000,
            analysis={"rhythm": {"bpm": 120}},
        )
        provider = CachedAnalysisProvider()
        result = provider.get_analysis("rhythm", str(track.id))
        assert result == {"bpm": 120}

    def test_get_analysis_raises_on_missing(self):
        """get_analysis raises KeyError when analysis missing."""
        track = Track.objects.create(
            original_filename="test.wav",
            storage_path="test.wav",
            file_size=1000,
            analysis={},
        )
        provider = CachedAnalysisProvider()
        with pytest.raises(KeyError):
            provider.get_analysis("rhythm", str(track.id))

    def test_get_analysis_raises_on_error(self):
        """get_analysis raises ValueError when analysis has error."""
        track = Track.objects.create(
            original_filename="test.wav",
            storage_path="test.wav",
            file_size=1000,
            analysis={"rhythm": {"error": "Failed"}},
        )
        provider = CachedAnalysisProvider()
        with pytest.raises(ValueError, match="Failed"):
            provider.get_analysis("rhythm", str(track.id))

    def test_has_analysis_true_when_present(self):
        """has_analysis returns True when analysis exists."""
        track = Track.objects.create(
            original_filename="test.wav",
            storage_path="test.wav",
            file_size=1000,
            analysis={"rhythm": {"bpm": 120}},
        )
        provider = CachedAnalysisProvider()
        assert provider.has_analysis("rhythm", str(track.id)) is True

    def test_has_analysis_false_when_missing(self):
        """has_analysis returns False when analysis missing."""
        track = Track.objects.create(
            original_filename="test.wav",
            storage_path="test.wav",
            file_size=1000,
            analysis={},
        )
        provider = CachedAnalysisProvider()
        assert provider.has_analysis("rhythm", str(track.id)) is False

    def test_has_analysis_false_for_nonexistent_track(self):
        """has_analysis returns False for nonexistent track."""
        provider = CachedAnalysisProvider()
        assert provider.has_analysis("rhythm", "00000000-0000-0000-0000-000000000000") is False
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_orchestrator/test_providers.py -v`
Expected: FAIL - module doesn't exist

**Step 3: Create providers module**

Create `src/orchestrator/providers.py`:

```python
"""
Analysis providers for the strategy pattern.

AnalysisProvider is an abstract base class defining how to fetch analysis data.
Implementations can query cached data from the database or run live analysis.

Note: Python's typing.Protocol offers an alternative approach using structural
typing (duck typing) rather than inheritance. With Protocol, any class with
matching method signatures is accepted, without needing to explicitly inherit.
"""

from abc import ABC, abstractmethod


class AnalysisProvider(ABC):
    """
    Abstract base class for analysis data providers.

    Subclasses implement different strategies for fetching analysis:
    - CachedAnalysisProvider: queries pre-computed results from database
    - LiveAnalysisProvider: runs agents on-demand
    - CacheFirstProvider: checks cache, falls back to live
    """

    @abstractmethod
    def get_analysis(self, agent_name: str, track_id: str) -> dict:
        """
        Get analysis result for an agent and track.

        Args:
            agent_name: Name of the agent (e.g., "rhythm", "spectral")
            track_id: UUID of the track

        Returns:
            Analysis data dictionary

        Raises:
            KeyError: If analysis is not available
            ValueError: If analysis contains an error
        """
        ...

    @abstractmethod
    def has_analysis(self, agent_name: str, track_id: str) -> bool:
        """
        Check if analysis is available.

        Args:
            agent_name: Name of the agent
            track_id: UUID of the track

        Returns:
            True if analysis exists and has no error
        """
        ...


class CachedAnalysisProvider(AnalysisProvider):
    """
    Provider that queries pre-computed analysis from the database.

    Uses Track.get_analysis() which was pre-computed by Celery workers.
    """

    def get_analysis(self, agent_name: str, track_id: str) -> dict:
        from src.api.models import Track
        track = Track.objects.get(id=track_id)
        return track.get_analysis(agent_name)

    def has_analysis(self, agent_name: str, track_id: str) -> bool:
        from src.api.models import Track
        try:
            track = Track.objects.get(id=track_id)
            return track.has_analysis(agent_name)
        except Track.DoesNotExist:
            return False
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_orchestrator/test_providers.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/orchestrator/providers.py tests/test_orchestrator/test_providers.py
git commit -m "feat(orchestrator): add AnalysisProvider ABC and CachedProvider"
```

---

### Task 7: Create BaseRouter ABC

**Files:**
- Create: `src/orchestrator/base.py`
- Test: `tests/test_orchestrator/test_base_router.py`

**Step 1: Write the failing test**

```python
# tests/test_orchestrator/test_base_router.py
import pytest
from unittest.mock import Mock, AsyncMock
from src.orchestrator.base import BaseRouter
from src.orchestrator.providers import AnalysisProvider
from src.orchestrator.events import RouterEvent


class TestBaseRouter:
    """Tests for BaseRouter ABC."""

    def test_cannot_instantiate_directly(self):
        """BaseRouter cannot be instantiated directly."""
        provider = Mock(spec=AnalysisProvider)
        with pytest.raises(TypeError):
            BaseRouter(provider=provider)

    def test_subclass_must_implement_process_question_stream(self):
        """Subclass must implement process_question_stream."""
        provider = Mock(spec=AnalysisProvider)

        class IncompleteRouter(BaseRouter):
            pass

        with pytest.raises(TypeError):
            IncompleteRouter(provider=provider)

    @pytest.mark.asyncio
    async def test_process_question_collects_events(self):
        """process_question collects events and returns result."""
        provider = Mock(spec=AnalysisProvider)

        class TestRouter(BaseRouter):
            async def process_question_stream(self, question, track_id):
                yield {"type": "tool_call", "tool": "analyse_rhythm"}
                yield {"type": "tool_result", "tool": "analyse_rhythm", "success": True}
                yield {"type": "token", "text": "The "}
                yield {"type": "token", "text": "tempo"}
                yield {"type": "done", "full_response": "The tempo is 120 BPM"}

        router = TestRouter(provider=provider, enable_tracing=False)
        result = await router.process_question("What's the tempo?", "track-123")

        assert result["response"] == "The tempo is 120 BPM"
        assert "analyse_rhythm" in result["tools_called"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_orchestrator/test_base_router.py -v`
Expected: FAIL - module doesn't exist

**Step 3: Create base router module**

Create `src/orchestrator/base.py`:

```python
"""
Base router class for LLM routing.

All router implementations inherit from BaseRouter and implement
the process_question_stream method for async event streaming.
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator

from .providers import AnalysisProvider
from .events import RouterEvent


class BaseRouter(ABC):
    """
    Abstract base class for LLM routers.

    Routers connect user questions to analysis tools via an LLM.
    They yield events during processing for real-time streaming.

    All routers accept an AnalysisProvider for fetching analysis data,
    allowing the same router to work with cached or live data.
    """

    def __init__(
        self,
        provider: AnalysisProvider,
        enable_tracing: bool = True,
        project_name: str = "audio-analysis-agents",
    ):
        """
        Initialize the router.

        Args:
            provider: Strategy for fetching analysis data
            enable_tracing: Whether to enable Opik tracing
            project_name: Project name for Opik
        """
        self.provider = provider
        self.enable_tracing = enable_tracing
        self.project_name = project_name

        if enable_tracing:
            from src.observability.tracing import init_tracing
            init_tracing(project_name)

    @abstractmethod
    async def process_question_stream(
        self,
        question: str,
        track_id: str,
    ) -> AsyncIterator[RouterEvent]:
        """
        Process a question, yielding events as they occur.

        Args:
            question: The user's question
            track_id: UUID of the track to analyse

        Yields:
            RouterEvent dictionaries (tool_call, tool_result, token, done, error)
        """
        ...

    async def process_question(self, question: str, track_id: str) -> dict:
        """
        Process a question in batch mode.

        Collects all events and returns the final result.
        Useful for non-streaming contexts like tests or CLI.

        Args:
            question: The user's question
            track_id: UUID of the track to analyse

        Returns:
            Dict with 'response' and 'tools_called'
        """
        tools_called = []
        full_response = ""

        async for event in self.process_question_stream(question, track_id):
            if event["type"] == "tool_call":
                tools_called.append(event["tool"])
            elif event["type"] == "token":
                full_response += event["text"]
            elif event["type"] == "done":
                full_response = event["full_response"]
            elif event["type"] == "error":
                return {"error": event["message"], "tools_called": tools_called}

        return {
            "response": full_response,
            "tools_called": tools_called,
        }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_orchestrator/test_base_router.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/orchestrator/base.py tests/test_orchestrator/test_base_router.py
git commit -m "feat(orchestrator): add BaseRouter ABC"
```

---

### Task 8: Implement Async LLMRouterLiteLLM

**Files:**
- Modify: `src/orchestrator/llm_routing_litellm.py`
- Test: `tests/test_orchestrator/test_llm_routing_litellm.py`

**Step 1: Write the failing test**

```python
# tests/test_orchestrator/test_llm_routing_litellm.py
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from src.orchestrator.llm_routing_litellm import LLMRouterLiteLLM
from src.orchestrator.providers import AnalysisProvider


class TestLLMRouterLiteLLM:
    """Tests for LLMRouterLiteLLM."""

    @pytest.mark.asyncio
    async def test_yields_tool_call_event(self):
        """Router yields tool_call event when LLM requests tool."""
        provider = Mock(spec=AnalysisProvider)
        provider.get_analysis = Mock(return_value={"bpm": 120})

        # Mock LiteLLM response with tool call
        mock_tool_response = MagicMock()
        mock_tool_response.choices = [MagicMock()]
        mock_tool_response.choices[0].message.tool_calls = [
            MagicMock(
                id="call_123",
                function=MagicMock(name="analyse_rhythm", arguments='{"track_id": "track-1"}')
            )
        ]
        mock_tool_response.choices[0].message.content = None

        # Mock streaming response
        async def mock_stream():
            yield MagicMock(choices=[MagicMock(delta=MagicMock(content="Done"))])

        mock_final_response = MagicMock()
        mock_final_response.choices = [MagicMock()]
        mock_final_response.choices[0].message.tool_calls = None
        mock_final_response.choices[0].message.content = "The tempo is 120 BPM"

        with patch("src.orchestrator.llm_routing_litellm.litellm") as mock_litellm:
            # First call returns tool request, second streams response
            mock_litellm.acompletion = AsyncMock(side_effect=[
                mock_tool_response,
                mock_stream(),
            ])

            router = LLMRouterLiteLLM(provider=provider, model="test-model", enable_tracing=False)

            events = []
            async for event in router.process_question_stream("What's the tempo?", "track-1"):
                events.append(event)

            # Should have tool_call, tool_result, tokens, done
            event_types = [e["type"] for e in events]
            assert "tool_call" in event_types
            assert "tool_result" in event_types

    @pytest.mark.asyncio
    async def test_uses_provider_for_analysis(self):
        """Router uses provider.get_analysis for tool results."""
        provider = Mock(spec=AnalysisProvider)
        provider.get_analysis = Mock(return_value={"bpm": 120})

        mock_tool_response = MagicMock()
        mock_tool_response.choices = [MagicMock()]
        mock_tool_response.choices[0].message.tool_calls = [
            MagicMock(
                id="call_123",
                function=MagicMock(name="analyse_rhythm", arguments='{"track_id": "track-1"}')
            )
        ]
        mock_tool_response.choices[0].message.content = None

        async def mock_stream():
            yield MagicMock(choices=[MagicMock(delta=MagicMock(content="Done"))])

        with patch("src.orchestrator.llm_routing_litellm.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=[
                mock_tool_response,
                mock_stream(),
            ])

            router = LLMRouterLiteLLM(provider=provider, model="test-model", enable_tracing=False)

            events = []
            async for event in router.process_question_stream("What's the tempo?", "track-1"):
                events.append(event)

            # Provider should have been called
            provider.get_analysis.assert_called_with("rhythm", "track-1")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_orchestrator/test_llm_routing_litellm.py -v`
Expected: FAIL - new implementation doesn't exist

**Step 3: Read existing llm_routing_litellm.py**

Read `src/orchestrator/llm_routing_litellm.py` to understand current structure.

**Step 4: Rewrite with async and BaseRouter**

Replace `src/orchestrator/llm_routing_litellm.py`:

```python
"""
LLM Routing Module - LiteLLM Provider-Agnostic Implementation

Uses LiteLLM for provider abstraction - same code works with Claude, Gemini, GPT-4.
Async implementation with streaming for real-time WebSocket updates.

Usage:
    from src.orchestrator.llm_routing_litellm import LLMRouterLiteLLM
    from src.orchestrator.providers import CachedAnalysisProvider

    provider = CachedAnalysisProvider()
    router = LLMRouterLiteLLM(provider=provider, model="gemini/gemini-2.0-flash")

    async for event in router.process_question_stream("What's the tempo?", track_id):
        print(event)
"""

import json
from typing import AsyncIterator

import litellm
from asgiref.sync import sync_to_async
from dotenv import load_dotenv

from .base import BaseRouter
from .events import RouterEvent
from .providers import AnalysisProvider
from src.agents.registry import get_tool_schemas_openai
from src.observability.tracing import trace_llm_call, trace_tool_execution

load_dotenv()


class LLMRouterLiteLLM(BaseRouter):
    """
    Async LLM router using LiteLLM for provider abstraction.

    Supports Claude, Gemini, GPT-4, and other providers through LiteLLM.
    Uses the AnalysisProvider strategy for fetching analysis data.
    """

    def __init__(
        self,
        provider: AnalysisProvider,
        model: str = "gemini/gemini-2.0-flash",
        **kwargs,
    ):
        """
        Initialize the LiteLLM router.

        Args:
            provider: Strategy for fetching analysis data
            model: LiteLLM model identifier. Examples:
                - "claude-sonnet-4-20250514" (Anthropic)
                - "gemini/gemini-2.0-flash" (Google)
                - "gpt-4o" (OpenAI)
            **kwargs: Passed to BaseRouter (enable_tracing, project_name)
        """
        super().__init__(provider=provider, **kwargs)
        self.model = model

    def _build_prompt(self, question: str, track_id: str) -> str:
        """Build the system prompt for the LLM."""
        return f"""You have access to audio analysis tools. The user is asking about an audio file.

Track ID: {track_id}

User question: {question}

Use the appropriate analysis tool(s) to answer this question. After receiving the analysis results, provide a clear, natural language response to the user's question."""

    async def process_question_stream(
        self,
        question: str,
        track_id: str,
    ) -> AsyncIterator[RouterEvent]:
        """
        Process a question, yielding events as they occur.

        Args:
            question: The user's question
            track_id: UUID of the track to analyse

        Yields:
            RouterEvent dictionaries
        """
        messages = [{"role": "user", "content": self._build_prompt(question, track_id)}]
        tools = get_tool_schemas_openai()

        with trace_llm_call(self.model, question, tools=[t["function"]["name"] for t in tools]):
            while True:
                response = await litellm.acompletion(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                )

                msg = response.choices[0].message

                if msg.tool_calls:
                    # Add assistant message to history
                    messages.append(msg.model_dump())

                    for tc in msg.tool_calls:
                        tool_name = tc.function.name
                        agent_name = tool_name.replace("analyse_", "")

                        yield {"type": "tool_call", "tool": tool_name}

                        with trace_tool_execution(agent_name, track_id):
                            try:
                                result = await sync_to_async(self.provider.get_analysis)(
                                    agent_name, track_id
                                )
                                tool_result = {"success": True, "data": result}
                                yield {"type": "tool_result", "tool": tool_name, "success": True}
                            except Exception as e:
                                tool_result = {"success": False, "error": str(e)}
                                yield {"type": "tool_result", "tool": tool_name, "success": False}

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(tool_result),
                        })

                else:
                    # Stream final response
                    full_response = ""

                    stream = await litellm.acompletion(
                        model=self.model,
                        messages=messages,
                        stream=True,
                    )

                    async for chunk in stream:
                        if chunk.choices[0].delta.content:
                            token = chunk.choices[0].delta.content
                            full_response += token
                            yield {"type": "token", "text": token}

                    yield {"type": "done", "full_response": full_response}
                    return
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_orchestrator/test_llm_routing_litellm.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/orchestrator/llm_routing_litellm.py tests/test_orchestrator/test_llm_routing_litellm.py
git commit -m "feat(orchestrator): rewrite LLMRouterLiteLLM with async streaming"
```

---

### Task 9: Update WebSocket Consumer

**Files:**
- Modify: `src/api/consumers.py`
- Test: `tests/test_api/test_consumers.py`

**Step 1: Write the failing test**

```python
# tests/test_api/test_consumers.py
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from channels.testing import WebsocketCommunicator
from django.test import TestCase
from src.api.consumers import ChatConsumer
from src.api.models import Track


class TestChatConsumer(TestCase):
    """Tests for ChatConsumer."""

    @pytest.mark.asyncio
    @pytest.mark.django_db(transaction=True)
    async def test_process_question_streams_events(self):
        """process_question streams router events to client."""
        # Create a ready track
        track = Track.objects.create(
            original_filename="test.wav",
            storage_path="test.wav",
            file_size=1000,
            status=Track.Status.READY,
            analysis={"rhythm": {"bpm": 120}},
        )

        # Mock router to yield events
        async def mock_stream(question, track_id):
            yield {"type": "tool_call", "tool": "analyse_rhythm"}
            yield {"type": "tool_result", "tool": "analyse_rhythm", "success": True}
            yield {"type": "token", "text": "The tempo"}
            yield {"type": "done", "full_response": "The tempo is 120 BPM"}

        mock_router = MagicMock()
        mock_router.process_question_stream = mock_stream

        with patch("src.api.consumers.LLMRouterLiteLLM", return_value=mock_router):
            with patch("src.api.consumers.CachedAnalysisProvider"):
                consumer = ChatConsumer()
                consumer.track = track
                consumer.track_id = str(track.id)
                consumer.send_json = AsyncMock()

                await consumer.process_question("What's the tempo?")

                # Should have sent multiple events
                assert consumer.send_json.call_count >= 4

                # Check event types
                calls = consumer.send_json.call_args_list
                event_types = [call[0][0]["type"] for call in calls]
                assert "tool_call" in event_types
                assert "done" in event_types
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_api/test_consumers.py::TestChatConsumer -v`
Expected: FAIL - consumer not updated

**Step 3: Read existing consumers.py**

Read `src/api/consumers.py` to understand current structure.

**Step 4: Update consumer to use async router**

Update the `process_question` method in `src/api/consumers.py`:

```python
"""
WebSocket Consumer for Streaming Chat

Handles real-time chat with streaming LLM responses.

Protocol:
1. Client connects to /ws/chat/{track_id}/?token=<jwt> OR ?api_key=xxx
2. Server sends status updates while analysis is in progress
3. Client sends: {"type": "question", "text": "What's the tempo?"}
4. Server streams back events from the router
"""

import json
import logging
from urllib.parse import parse_qs

from channels.generic.websocket import AsyncWebsocketConsumer
from django.conf import settings
from rest_framework_simplejwt.exceptions import InvalidToken, TokenError
from rest_framework_simplejwt.tokens import AccessToken

from .auth import check_api_key
from .models import Track

logger = logging.getLogger(__name__)


class ChatConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer for streaming audio analysis chat.

    Uses LLMRouterLiteLLM with CachedAnalysisProvider for instant responses.
    """

    async def connect(self):
        """Handle WebSocket connection."""
        self.track_id = self.scope["url_route"]["kwargs"]["track_id"]
        self.is_demo = False
        self.track = None

        # Parse query string
        query_string = self.scope.get("query_string", b"").decode()
        params = parse_qs(query_string)

        # Try JWT auth first
        token = params.get("token", [None])[0]
        if token:
            try:
                access_token = AccessToken(token)
                self.is_demo = access_token.get("is_demo", False)
            except (InvalidToken, TokenError):
                await self.close(code=4001)
                return
        else:
            # Fall back to API key
            api_key = params.get("api_key", [None])[0]
            if not api_key or not check_api_key(api_key):
                await self.close(code=4001)
                return

        # Check track exists
        from asgiref.sync import sync_to_async
        self.track = await sync_to_async(Track.get_or_none)(self.track_id)
        if not self.track:
            await self.close(code=4004)
            return

        # Join channel group for status updates
        self.group_name = f"track_{self.track_id}"
        await self.channel_layer.group_add(
            self.group_name,
            self.channel_name
        )

        await self.accept()
        await self.send_current_status()

    async def disconnect(self, close_code):
        """Handle WebSocket disconnection."""
        if hasattr(self, 'group_name'):
            await self.channel_layer.group_discard(
                self.group_name,
                self.channel_name
            )

    async def send_current_status(self):
        """Send the current track status to the client."""
        from asgiref.sync import sync_to_async

        self.track = await sync_to_async(Track.get_or_none)(self.track_id)
        if not self.track:
            return

        await self.send_json({
            "type": "status",
            "status": self.track.status,
            "message": self.track.status_message or self._status_message(self.track.status),
        })

    def _status_message(self, status: str) -> str:
        """Get default message for a status."""
        messages = {
            Track.Status.PENDING: "Waiting to start analysis...",
            Track.Status.ANALYZING: "Analyzing audio...",
            Track.Status.READY: "Ready for questions",
            Track.Status.FAILED: "Analysis failed",
        }
        return messages.get(status, "")

    # Handlers for Celery notifications
    async def track_status(self, event):
        """Handle track status update from Celery."""
        await self.send_json({
            "type": "status",
            "status": event["status"],
            "message": event["message"],
        })

    async def track_progress(self, event):
        """Handle track progress update from Celery."""
        await self.send_json({
            "type": "thinking",
            "message": f"{event['agent'].title()} analysis: {event['state']}",
        })

    async def receive(self, text_data):
        """Handle incoming message from client."""
        logger.info(f"Received message: {text_data[:200]}")
        try:
            data = json.loads(text_data)
        except json.JSONDecodeError:
            await self.send_error("Invalid JSON")
            return

        msg_type = data.get("type")

        if msg_type == "question":
            question = data.get("text", "").strip()
            if not question:
                await self.send_error("Empty question")
                return

            logger.info(f"Processing question: {question}")
            await self.process_question(question)

        else:
            await self.send_error(f"Unknown message type: {msg_type}")

    async def process_question(self, question: str):
        """
        Process a question using the async router.

        Streams events directly to the WebSocket client.
        """
        from src.orchestrator.providers import CachedAnalysisProvider
        from src.orchestrator.llm_routing_litellm import LLMRouterLiteLLM
        from asgiref.sync import sync_to_async

        # Refresh track
        self.track = await sync_to_async(Track.get_or_none)(self.track_id)
        if not self.track:
            await self.send_error("Track not found")
            return

        if not self.track.is_ready:
            await self.send_error(f"Track not ready. Status: {self.track.status}")
            return

        # Create router with cached provider
        provider = CachedAnalysisProvider()
        model = getattr(settings, "LLM_MODEL", "gemini/gemini-2.0-flash")
        router = LLMRouterLiteLLM(provider=provider, model=model, enable_tracing=True)

        # Stream events to client
        try:
            async for event in router.process_question_stream(question, str(self.track.id)):
                await self.send_json(event)
        except Exception as e:
            logger.exception(f"Error processing question: {e}")
            await self.send_error(str(e))

    async def send_json(self, data: dict):
        """Send JSON message to client."""
        await self.send(text_data=json.dumps(data))

    async def send_error(self, message: str):
        """Send error message to client."""
        await self.send_json({
            "type": "error",
            "message": message,
        })
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_api/test_consumers.py::TestChatConsumer -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/api/consumers.py tests/test_api/test_consumers.py
git commit -m "feat(api): update WebSocket consumer to use async router"
```

---

### Task 10: Update Views to Use Fat Model

**Files:**
- Modify: `src/api/views.py`
- Test: `tests/test_api/test_views.py`

**Step 1: Write the failing test**

```python
# tests/test_api/test_views.py (add to existing)
import pytest
from django.test import TestCase
from rest_framework.test import APIClient
from src.api.models import Track


class TestTrackUploadViewUsesModel(TestCase):
    """Tests that views use Track model methods."""

    def test_upload_uses_queue_analysis(self):
        """Upload should use track.queue_analysis()."""
        # This is more of an integration test
        # The key is that views.py no longer has get_track or get_track_path functions
        pass


class TestUtilityFunctionsRemoved(TestCase):
    """Tests that utility functions were moved to model."""

    def test_get_track_not_in_views(self):
        """get_track function should not exist in views."""
        from src.api import views
        assert not hasattr(views, 'get_track'), "get_track should be moved to Track.get_or_none"

    def test_get_track_path_not_in_views(self):
        """get_track_path function should not exist in views."""
        from src.api import views
        assert not hasattr(views, 'get_track_path'), "get_track_path should be moved to Track.file_path"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_api/test_views.py::TestUtilityFunctionsRemoved -v`
Expected: FAIL - functions still exist in views

**Step 3: Read existing views.py**

Read `src/api/views.py` to see current utility functions.

**Step 4: Remove utility functions and use model methods**

Update `src/api/views.py` to remove `get_track` and `get_track_path`, use `Track.get_or_none` and `track.queue_analysis()`:

```python
"""
API Views for audio upload and track management.
"""

from pathlib import Path

from django.core.files.storage import default_storage
from rest_framework import status
from rest_framework.parsers import MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.authentication import JWTAuthentication

from .auth import APIKeyAuthentication
from .models import Track


class TrackUploadView(APIView):
    """
    Upload an audio file.

    POST /api/tracks/
    """

    authentication_classes = [JWTAuthentication, APIKeyAuthentication]
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser]

    def post(self, request):
        if "file" not in request.FILES:
            return Response(
                {"error": "No file provided"},
                status=status.HTTP_400_BAD_REQUEST
            )

        uploaded_file = request.FILES["file"]

        # Validate file extension
        allowed_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
        ext = Path(uploaded_file.name).suffix.lower()
        if ext not in allowed_extensions:
            return Response(
                {"error": f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Save using Django's storage backend
        import uuid
        track_id = uuid.uuid4()
        filename = f"{track_id}{ext}"
        saved_path = default_storage.save(filename, uploaded_file)
        file_url = default_storage.url(saved_path)

        # Create Track in database
        track = Track.objects.create(
            id=track_id,
            original_filename=uploaded_file.name,
            storage_path=saved_path,
            file_url=file_url,
            file_size=uploaded_file.size,
            user=request.user if request.user.is_authenticated else None,
        )

        # Queue background analysis using model method
        task_id = track.queue_analysis()

        response_data = {
            "track_id": str(track.id),
            "filename": uploaded_file.name,
            "status": track.status,
        }
        if task_id:
            response_data["task_id"] = task_id

        return Response(response_data, status=status.HTTP_201_CREATED)


class TrackDetailView(APIView):
    """
    Get or delete a track.

    GET /api/tracks/{track_id}/
    DELETE /api/tracks/{track_id}/
    """

    authentication_classes = [JWTAuthentication, APIKeyAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request, track_id):
        track = Track.get_or_none(track_id)
        if not track:
            return Response(
                {"error": "Track not found"},
                status=status.HTTP_404_NOT_FOUND
            )

        return Response({
            "track_id": str(track.id),
            "original_filename": track.original_filename,
            "storage_path": track.storage_path,
            "url": track.file_url,
            "size": track.file_size,
            "duration": track.duration,
            "status": track.status,
            "status_message": track.status_message,
            "created_at": track.created_at.isoformat(),
            "analyzed_at": track.analyzed_at.isoformat() if track.analyzed_at else None,
        })

    def delete(self, request, track_id):
        track = Track.get_or_none(track_id)
        if not track:
            return Response(
                {"error": "Track not found"},
                status=status.HTTP_404_NOT_FOUND
            )

        # Delete file using storage backend
        if default_storage.exists(track.storage_path):
            default_storage.delete(track.storage_path)

        track.delete()

        return Response(status=status.HTTP_204_NO_CONTENT)


class TrackListView(APIView):
    """
    List all tracks.

    GET /api/tracks/list/
    """

    authentication_classes = [JWTAuthentication, APIKeyAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        if request.user.is_authenticated:
            tracks = Track.objects.filter(user=request.user)
        else:
            tracks = Track.objects.none()

        return Response([
            {
                "track_id": str(t.id),
                "original_filename": t.original_filename,
                "status": t.status,
                "duration": t.duration,
                "created_at": t.created_at.isoformat(),
            }
            for t in tracks
        ])
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_api/test_views.py::TestUtilityFunctionsRemoved -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/api/views.py tests/test_api/test_views.py
git commit -m "refactor(api): use fat model methods, remove utility functions from views"
```

---

### Task 11: Run Full Test Suite

**Step 1: Run all tests**

```bash
pytest tests/ -v --tb=short
```

Expected: All tests PASS

**Step 2: Fix any failing tests**

If tests fail, fix issues and re-run.

**Step 3: Commit any fixes**

```bash
git add -A
git commit -m "fix: address test failures from refactor"
```

---

### Task 12: Update Imports and Exports

**Files:**
- Modify: `src/orchestrator/__init__.py`
- Modify: `src/agents/__init__.py`

**Step 1: Update orchestrator __init__.py**

```python
# src/orchestrator/__init__.py
"""
Orchestrator module for LLM routing.

Provides routers that connect user questions to analysis tools via LLMs.
"""

from .events import (
    RouterEvent,
    ToolCallEvent,
    ToolResultEvent,
    TokenEvent,
    DoneEvent,
    ErrorEvent,
)
from .providers import (
    AnalysisProvider,
    CachedAnalysisProvider,
)
from .base import BaseRouter
from .llm_routing_litellm import LLMRouterLiteLLM

__all__ = [
    # Events
    "RouterEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "TokenEvent",
    "DoneEvent",
    "ErrorEvent",
    # Providers
    "AnalysisProvider",
    "CachedAnalysisProvider",
    # Routers
    "BaseRouter",
    "LLMRouterLiteLLM",
]
```

**Step 2: Update agents __init__.py to export registry functions**

```python
# src/agents/__init__.py (add to existing exports)
from .registry import (
    AGENT_REGISTRY,
    register_agent,
    get_all_agents,
    get_agent,
    get_tool_schemas_openai,
)
```

**Step 3: Commit**

```bash
git add src/orchestrator/__init__.py src/agents/__init__.py
git commit -m "chore: update module exports"
```

---

### Task 13: Final Integration Test

**Step 1: Write integration test**

```python
# tests/test_integration/test_router_integration.py
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from django.test import TestCase
from src.api.models import Track
from src.orchestrator.providers import CachedAnalysisProvider
from src.orchestrator.llm_routing_litellm import LLMRouterLiteLLM


class TestRouterIntegration(TestCase):
    """Integration tests for the full router flow."""

    @pytest.mark.asyncio
    async def test_full_flow_with_cached_data(self):
        """Test router with real cached data from database."""
        # Create track with analysis
        track = Track.objects.create(
            original_filename="test.wav",
            storage_path="test.wav",
            file_size=1000,
            status=Track.Status.READY,
            analysis={
                "rhythm": {"bpm": 120, "beat_count": 48},
                "spectral": {"centroid": 2000},
            },
        )

        # Mock LiteLLM
        mock_tool_response = MagicMock()
        mock_tool_response.choices = [MagicMock()]
        mock_tool_response.choices[0].message.tool_calls = [
            MagicMock(
                id="call_123",
                function=MagicMock(
                    name="analyse_rhythm",
                    arguments=f'{{"track_id": "{track.id}"}}'
                )
            )
        ]
        mock_tool_response.choices[0].message.content = None

        async def mock_stream():
            yield MagicMock(choices=[MagicMock(delta=MagicMock(content="120 BPM"))])

        with patch("src.orchestrator.llm_routing_litellm.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=[
                mock_tool_response,
                mock_stream(),
            ])

            provider = CachedAnalysisProvider()
            router = LLMRouterLiteLLM(
                provider=provider,
                model="test-model",
                enable_tracing=False
            )

            events = []
            async for event in router.process_question_stream(
                "What's the tempo?",
                str(track.id)
            ):
                events.append(event)

            # Verify tool_result got real data from DB
            tool_results = [e for e in events if e["type"] == "tool_result"]
            assert len(tool_results) == 1
            assert tool_results[0]["success"] is True
```

**Step 2: Run integration test**

```bash
pytest tests/test_integration/test_router_integration.py -v
```

Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_integration/test_router_integration.py
git commit -m "test: add router integration test"
```

---

## Summary

**Tasks completed:**
1. Add description field to BaseAgent
2. Add descriptions to existing agents
3. Add tool schema generation to registry
4. Add fat model methods to Track
5. Create router events
6. Create AnalysisProvider ABC and CachedProvider
7. Create BaseRouter ABC
8. Implement async LLMRouterLiteLLM
9. Update WebSocket consumer
10. Update views to use fat model
11. Run full test suite
12. Update imports and exports
13. Final integration test

**Files created:**
- `src/orchestrator/events.py`
- `src/orchestrator/providers.py`
- `src/orchestrator/base.py`
- `tests/test_agents/test_base.py`
- `tests/test_agents/test_agent_descriptions.py`
- `tests/test_orchestrator/test_events.py`
- `tests/test_orchestrator/test_providers.py`
- `tests/test_orchestrator/test_base_router.py`
- `tests/test_orchestrator/test_llm_routing_litellm.py`
- `tests/test_api/test_models.py`
- `tests/test_api/test_consumers.py`
- `tests/test_integration/test_router_integration.py`

**Files modified:**
- `src/agents/base.py`
- `src/agents/spectral.py`
- `src/agents/temporal.py`
- `src/agents/rhythm.py`
- `src/agents/registry.py`
- `src/agents/__init__.py`
- `src/api/models.py`
- `src/api/views.py`
- `src/api/consumers.py`
- `src/orchestrator/llm_routing_litellm.py`
- `src/orchestrator/__init__.py`
