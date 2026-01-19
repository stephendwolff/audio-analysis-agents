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
