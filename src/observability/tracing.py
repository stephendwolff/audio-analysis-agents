"""
Opik Tracing Setup

Provides tracing for:
- LLM calls (prompts, responses, tokens, latency)
- Tool/agent executions (which agent, duration, success/fail)

Usage:
    from src.observability import init_tracing, trace_llm_call, trace_tool_execution

    init_tracing()  # Call once at startup

    with trace_llm_call("gemini-2.0-flash", prompt="...") as span:
        response = client.generate(...)
        span.set_output(response)

    with trace_tool_execution("rhythm", track_id="...") as span:
        result = agent.analyse(...)
        span.set_output(result)
"""

import os
from contextlib import contextmanager
from functools import wraps
from typing import Any

import opik
from dotenv import load_dotenv

load_dotenv()

# Global client reference
_client: opik.Opik | None = None


def init_tracing(project_name: str = "audio-analysis-agents") -> opik.Opik:
    """
    Initialise Opik tracing.

    Call this once at application startup.

    Args:
        project_name: Name of the project in Opik dashboard

    Returns:
        Opik client instance
    """
    global _client

    api_key = os.getenv("OPIK_API_KEY")
    if not api_key:
        print("Warning: OPIK_API_KEY not set, tracing disabled")
        return None

    _client = opik.Opik(project_name=project_name)
    print(f"Opik tracing initialised for project: {project_name}")
    return _client


def get_client() -> opik.Opik | None:
    """Get the global Opik client."""
    return _client


@contextmanager
def trace_llm_call(
    model: str,
    prompt: str,
    tools: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
):
    """
    Context manager for tracing LLM calls.

    Args:
        model: Model name (e.g., "gemini-2.0-flash")
        prompt: The prompt sent to the LLM
        tools: List of tool names available to the LLM
        metadata: Additional metadata to attach

    Yields:
        Trace span that can be updated with output

    Example:
        with trace_llm_call("gemini-2.0-flash", prompt=user_question) as span:
            response = client.generate(...)
            span.update(output=response.text)
    """
    if _client is None:
        # Tracing disabled, yield a no-op object
        yield _NoOpSpan()
        return

    trace = _client.trace(
        name="llm_call",
        input={"prompt": prompt, "tools": tools or []},
        metadata={"model": model, **(metadata or {})},
    )

    try:
        yield trace
    except Exception as e:
        trace.update(metadata={"error": str(e)})
        raise
    finally:
        trace.end()


@contextmanager
def trace_tool_execution(
    tool_name: str,
    track_id: str,
    input_data: dict[str, Any] | None = None,
):
    """
    Context manager for tracing tool/agent executions.

    Args:
        tool_name: Name of the tool/agent (e.g., "rhythm", "spectral")
        track_id: ID of the audio track being analysed
        input_data: Additional input parameters

    Yields:
        Trace span that can be updated with output

    Example:
        with trace_tool_execution("rhythm", track_id="abc123") as span:
            result = rhythm_agent.analyse(samples, sr)
            span.update(output=result.model_dump())
    """
    if _client is None:
        yield _NoOpSpan()
        return

    trace = _client.trace(
        name=f"tool_{tool_name}",
        input={"track_id": track_id, **(input_data or {})},
        metadata={"tool": tool_name},
    )

    try:
        yield trace
    except Exception as e:
        trace.update(metadata={"error": str(e), "success": False})
        raise
    finally:
        trace.end()


def trace_function(name: str | None = None):
    """
    Decorator for tracing function calls.

    Args:
        name: Optional name override (defaults to function name)

    Example:
        @trace_function()
        def process_audio(track_id: str):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            trace_name = name or func.__name__

            if _client is None:
                return func(*args, **kwargs)

            trace = _client.trace(
                name=trace_name,
                input={"args": str(args), "kwargs": str(kwargs)},
            )

            try:
                result = func(*args, **kwargs)
                trace.update(output=str(result))
                return result
            except Exception as e:
                trace.update(metadata={"error": str(e)})
                raise
            finally:
                trace.end()

        return wrapper
    return decorator


class _NoOpSpan:
    """No-op span for when tracing is disabled."""

    def update(self, **kwargs):
        pass

    def end(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
