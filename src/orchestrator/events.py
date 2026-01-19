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
