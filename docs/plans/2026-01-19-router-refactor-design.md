# Router Refactor Design

## Overview

Refactor the LLM routing layer to eliminate code duplication, support real-time WebSocket streaming, and use pre-cached analysis results from the database.

## Goals

1. Remove duplicated tool definitions across three routers
2. Make analysis source pluggable (cached vs live)
3. Support real-time streaming to WebSocket clients
4. Clean up Django models (fat models, thin views)
5. Keep all three routers as learning examples

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌───────────────────┐
│   Frontend  │────▶│  ChatConsumer    │────▶│  LLMRouterLiteLLM │
│  WebSocket  │◀────│  (async for)     │◀────│  (AsyncIterator)  │
└─────────────┘     └──────────────────┘     └─────────┬─────────┘
                                                       │
                                                       ▼
                                             ┌───────────────────┐
                                             │ AnalysisProvider  │
                                             │  (strategy)       │
                                             └─────────┬─────────┘
                                                       │
                            ┌──────────────────────────┼──────────────────────────┐
                            ▼                          ▼                          ▼
                    ┌───────────────┐         ┌───────────────┐         ┌─────────────────┐
                    │CachedProvider │         │ LiveProvider  │         │CacheFirstProvider│
                    │  (query DB)   │         │ (run agents)  │         │ (cache + live)  │
                    └───────────────┘         └───────────────┘         └─────────────────┘
```

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Primary use case | WebSocket chat | Users interact via web UI |
| Router backends | Keep all 3 | Learning project - show different approaches |
| Analysis source | Strategy pattern (ABC) | Clean separation, easy to test |
| Tool definitions | Generated from agent registry | Single source of truth |
| Agent metadata | name + description | YAGNI - all tools take track_id |
| Django models | Fat models, thin views | Django best practice |
| Streaming | Async router | Real-time WebSocket streaming |
| Testing | TDD | Tests first at each layer |

## Component Details

### 1. Agent Registry & Tool Generation

Agents expose `name` and `description`. Registry generates tool schemas for each LLM format.

```python
# src/agents/base.py
class BaseAgent(ABC):
    name: str
    description: str

    @abstractmethod
    def analyse(self, samples: np.ndarray, sample_rate: int) -> AnalysisResult: ...
```

```python
# src/agents/registry.py
AGENT_REGISTRY: dict[str, type[BaseAgent]] = {}

def register_agent(cls):
    AGENT_REGISTRY[cls.name] = cls
    return cls

def get_tool_schemas_openai() -> list[dict]:
    """Generate OpenAI-format tool schemas from registered agents."""
    return [
        {
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
        }
        for agent in (cls() for cls in AGENT_REGISTRY.values())
    ]

def get_tool_schemas_gemini() -> list:
    """Generate Gemini-format schemas from registered agents."""
    from google.genai import types
    return [
        types.FunctionDeclaration(
            name=f"analyse_{agent.name}",
            description=agent.description,
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "track_id": types.Schema(
                        type=types.Type.STRING,
                        description="Track ID to analyse"
                    )
                },
                required=["track_id"]
            )
        )
        for agent in (cls() for cls in AGENT_REGISTRY.values())
    ]
```

### 2. AnalysisProvider Strategy Pattern

Abstract base class with concrete implementations. Uses ABC for explicit inheritance (educational). Note: Python's `Protocol` offers structural typing as an alternative.

```python
# src/orchestrator/providers.py
from abc import ABC, abstractmethod

class AnalysisProvider(ABC):
    """
    Strategy for fetching analysis results.

    Note: Python's typing.Protocol offers an alternative approach using
    structural typing (duck typing) rather than inheritance. With Protocol,
    any class with matching method signatures is accepted, without needing
    to explicitly inherit from the base class.
    """

    @abstractmethod
    def get_analysis(self, agent_name: str, track_id: str) -> dict:
        """Return analysis result. Raises KeyError if unavailable."""
        ...

    @abstractmethod
    def has_analysis(self, agent_name: str, track_id: str) -> bool:
        """Check if analysis is available."""
        ...


class CachedAnalysisProvider(AnalysisProvider):
    """Queries pre-computed analysis from database."""

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


class LiveAnalysisProvider(AnalysisProvider):
    """Runs agents on-demand, optionally caching results."""

    def __init__(self, cache_results: bool = True):
        self.cache_results = cache_results

    def get_analysis(self, agent_name: str, track_id: str) -> dict:
        from src.api.models import Track
        from src.agents.registry import get_agent
        from src.tools.loader import load_audio

        track = Track.objects.get(id=track_id)
        agent = get_agent(agent_name)

        if not agent:
            raise KeyError(f"Unknown agent: {agent_name}")

        audio = load_audio(track.file_path, target_sr=22050, mono=True)
        result = agent.analyse(audio.samples, audio.sample_rate)

        if not result.success:
            raise ValueError(result.error)

        if self.cache_results:
            track.set_analysis(agent_name, result.data)

        return result.data

    def has_analysis(self, agent_name: str, track_id: str) -> bool:
        from src.agents.registry import get_agent
        return get_agent(agent_name) is not None


class CacheFirstProvider(AnalysisProvider):
    """Check cache first, fall back to live analysis."""

    def __init__(self):
        self.cached = CachedAnalysisProvider()
        self.live = LiveAnalysisProvider(cache_results=True)

    def get_analysis(self, agent_name: str, track_id: str) -> dict:
        if self.cached.has_analysis(agent_name, track_id):
            return self.cached.get_analysis(agent_name, track_id)
        return self.live.get_analysis(agent_name, track_id)

    def has_analysis(self, agent_name: str, track_id: str) -> bool:
        return (
            self.cached.has_analysis(agent_name, track_id) or
            self.live.has_analysis(agent_name, track_id)
        )
```

### 3. Track Model (Fat Model)

Move utility functions from views to model methods.

```python
# src/api/models.py
class Track(models.Model):
    # ... existing fields ...

    @classmethod
    def get_or_none(cls, track_id: str) -> "Track | None":
        """Get track by ID, returning None if not found."""
        try:
            return cls.objects.get(id=track_id)
        except cls.DoesNotExist:
            return None

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
```

### 4. Router Events

Typed events for streaming.

```python
# src/orchestrator/events.py
from typing import TypedDict, Literal

class ToolCallEvent(TypedDict):
    type: Literal["tool_call"]
    tool: str

class ToolResultEvent(TypedDict):
    type: Literal["tool_result"]
    tool: str
    success: bool

class TokenEvent(TypedDict):
    type: Literal["token"]
    text: str

class DoneEvent(TypedDict):
    type: Literal["done"]
    full_response: str

class ErrorEvent(TypedDict):
    type: Literal["error"]
    message: str

RouterEvent = ToolCallEvent | ToolResultEvent | TokenEvent | DoneEvent | ErrorEvent
```

### 5. Base Router

Abstract base class for all routers.

```python
# src/orchestrator/base.py
from abc import ABC, abstractmethod
from typing import AsyncIterator

from .providers import AnalysisProvider
from .events import RouterEvent

class BaseRouter(ABC):
    """
    Base class for LLM routers.

    All routers accept an AnalysisProvider and yield RouterEvents.
    """

    def __init__(
        self,
        provider: AnalysisProvider,
        enable_tracing: bool = True,
        project_name: str = "audio-analysis-agents",
    ):
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
        """Process question, yielding events as they occur."""
        ...

    async def process_question(self, question: str, track_id: str) -> dict:
        """Batch mode - collect all events and return final result."""
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
                return {"error": event["message"]}

        return {
            "response": full_response,
            "tools_called": tools_called,
        }
```

### 6. LiteLLM Router (Async)

Full async implementation with streaming.

```python
# src/orchestrator/llm_routing_litellm.py
import json
from typing import AsyncIterator

import litellm
from asgiref.sync import sync_to_async

from .base import BaseRouter
from .events import RouterEvent
from .providers import AnalysisProvider
from src.agents.registry import get_tool_schemas_openai
from src.observability.tracing import trace_llm_call, trace_tool_execution

class LLMRouterLiteLLM(BaseRouter):
    """
    Async LLM router using LiteLLM for provider abstraction.

    Supports Claude, Gemini, GPT-4, and other providers.
    """

    def __init__(
        self,
        provider: AnalysisProvider,
        model: str = "gemini/gemini-2.0-flash",
        **kwargs,
    ):
        super().__init__(provider=provider, **kwargs)
        self.model = model

    def _build_prompt(self, question: str, track_id: str) -> str:
        return f"""You have access to audio analysis tools. The user is asking about an audio file.

Track ID: {track_id}

User question: {question}

Use the appropriate analysis tool(s) to answer this question. After receiving the analysis results, provide a clear, natural language response to the user's question."""

    async def process_question_stream(
        self,
        question: str,
        track_id: str,
    ) -> AsyncIterator[RouterEvent]:
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

### 7. WebSocket Consumer

Simple async iteration over router events.

```python
# src/api/consumers.py (process_question method)
async def process_question(self, question: str):
    from src.orchestrator.providers import CachedAnalysisProvider
    from src.orchestrator.llm_routing_litellm import LLMRouterLiteLLM
    from src.api.models import Track
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
    router = LLMRouterLiteLLM(provider=provider, model=model)

    # Stream events to client
    try:
        async for event in router.process_question_stream(question, str(self.track.id)):
            await self.send_json(event)
    except Exception as e:
        await self.send_error(str(e))
```

## Testing Strategy

Tests written first at each layer:

**Layer 1: Agent Registry**
- `test_register_agent_adds_to_registry`
- `test_get_tool_schemas_openai_generates_from_agents`
- `test_get_tool_schemas_gemini_generates_from_agents`

**Layer 2: AnalysisProvider**
- `test_cached_provider_returns_analysis_from_db`
- `test_cached_provider_raises_on_missing_analysis`
- `test_live_provider_runs_agent_and_caches`
- `test_cache_first_provider_uses_cache_when_available`
- `test_cache_first_provider_falls_back_to_live`

**Layer 3: Track Model**
- `test_track_get_or_none_returns_track`
- `test_track_get_or_none_returns_none_for_missing`
- `test_track_file_path_returns_local_path`
- `test_track_file_path_returns_url_for_s3`
- `test_track_has_analysis_true_when_present`
- `test_track_has_analysis_false_when_error`
- `test_track_get_analysis_raises_on_missing`
- `test_track_queue_analysis_returns_task_id`

**Layer 4: Router**
- `test_router_yields_tool_call_events`
- `test_router_yields_tool_result_events`
- `test_router_yields_token_events`
- `test_router_uses_provider_for_analysis`
- `test_router_process_question_batch_mode`

**Layer 5: WebSocket Consumer**
- `test_consumer_streams_events_to_client`
- `test_consumer_rejects_if_track_not_ready`

## File Changes

**New files:**
- `src/orchestrator/events.py`
- `src/orchestrator/providers.py`
- `src/orchestrator/base.py`

**Modified files:**
- `src/agents/base.py` - Add description field
- `src/agents/spectral.py` - Add description
- `src/agents/temporal.py` - Add description
- `src/agents/rhythm.py` - Add description
- `src/agents/registry.py` - Add tool schema generation
- `src/orchestrator/llm_routing.py` - Inherit BaseRouter, use provider
- `src/orchestrator/llm_routing_litellm.py` - Async, inherit BaseRouter, use provider
- `src/orchestrator/llm_routing_langgraph.py` - Inherit BaseRouter, use provider
- `src/api/models.py` - Fat Track model
- `src/api/views.py` - Thin views, use model methods
- `src/api/consumers.py` - Use async router

## Implementation Order

1. Agent registry tool schema generation (+ tests)
2. Track model methods (+ tests)
3. AnalysisProvider classes (+ tests)
4. Router events and base class
5. LLMRouterLiteLLM async implementation (+ tests)
6. WebSocket consumer integration (+ tests)
7. Update Gemini router to match pattern
8. Update LangGraph router to match pattern
