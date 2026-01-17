"""
WebSocket Consumer for Streaming Chat

Handles real-time chat with streaming LLM responses.

Protocol:
1. Client connects to /ws/chat/{track_id}/?token=<jwt> OR ?api_key=xxx
2. Server sends status updates while analysis is in progress:
   - {"type": "status", "status": "analyzing", "message": "Starting analysis..."}
   - {"type": "thinking", "message": "Rhythm analysis: running"}
   - {"type": "status", "status": "ready", "message": "Analysis complete"}
3. Client sends: {"type": "question", "text": "What's the tempo?"}
4. Server streams back:
   - {"type": "tool_call", "tool": "analyse_rhythm"}
   - {"type": "tool_result", "tool": "analyse_rhythm", "success": true}
   - {"type": "token", "text": "The"}
   - {"type": "token", "text": " tempo"}
   - {"type": "done", "full_response": "The tempo is 120 BPM"}
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
from .views import get_track

logger = logging.getLogger(__name__)


class ChatConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer for streaming audio analysis chat.

    Accepts either:
    - ?token=<jwt> - JWT authentication
    - ?api_key=<key> - Legacy API key authentication

    Joins a channel group for the track to receive status updates from Celery.
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
                # JWT is valid
            except (InvalidToken, TokenError):
                await self.close(code=4001)  # Unauthorized
                return
        else:
            # Fall back to API key
            api_key = params.get("api_key", [None])[0]
            if not api_key or not check_api_key(api_key):
                await self.close(code=4001)  # Unauthorized
                return

        # Check track exists in database
        from asgiref.sync import sync_to_async
        self.track = await sync_to_async(get_track)(self.track_id)
        if not self.track:
            await self.close(code=4004)  # Not found
            return

        # Join channel group for this track (to receive Celery notifications)
        self.group_name = f"track_{self.track_id}"
        await self.channel_layer.group_add(
            self.group_name,
            self.channel_name
        )

        await self.accept()

        # Send current status to client
        await self.send_current_status()

    async def disconnect(self, close_code):
        """Handle WebSocket disconnection."""
        # Leave channel group
        if hasattr(self, 'group_name'):
            await self.channel_layer.group_discard(
                self.group_name,
                self.channel_name
            )

    async def send_current_status(self):
        """Send the current track status to the client."""
        from asgiref.sync import sync_to_async

        # Refresh track from database
        self.track = await sync_to_async(get_track)(self.track_id)
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

    # Handlers for Celery notifications (via channel layer)
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
        Process a question using cached analysis results.

        Queries the database for pre-computed analysis instead of running agents.
        """
        import litellm
        from asgiref.sync import sync_to_async

        # Refresh track from database
        self.track = await sync_to_async(get_track)(self.track_id)
        if not self.track:
            await self.send_error("Track not found")
            return

        # Check if track is ready
        if self.track.status != Track.Status.READY:
            await self.send_error(f"Track not ready for questions. Status: {self.track.status}")
            return

        # Tool definitions
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "analyse_spectral",
                    "description": "Analyse frequency content of audio. Use for questions about frequency, pitch, tone, timbre.",
                    "parameters": {
                        "type": "object",
                        "properties": {"track_id": {"type": "string"}},
                        "required": ["track_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyse_temporal",
                    "description": "Analyse time-domain properties. Use for questions about duration, volume, dynamics.",
                    "parameters": {
                        "type": "object",
                        "properties": {"track_id": {"type": "string"}},
                        "required": ["track_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyse_rhythm",
                    "description": "Analyse tempo and rhythm. Use for questions about tempo, BPM, beats.",
                    "parameters": {
                        "type": "object",
                        "properties": {"track_id": {"type": "string"}},
                        "required": ["track_id"]
                    }
                }
            },
        ]

        # Build prompt
        prompt = f"""You have access to audio analysis tools. The user is asking about an audio file.

Audio file: {self.track.original_filename}
Duration: {self.track.duration:.1f}s

User question: {question}

Use the appropriate analysis tool(s) to answer this question."""

        messages = [{"role": "user", "content": prompt}]
        model = getattr(settings, "LLM_MODEL", "gemini/gemini-2.0-flash")

        try:
            logger.info(f"Calling LLM: {model}")
            # First call - might return tool calls
            response = litellm.completion(
                model=model,
                messages=messages,
                tools=tools,
            )
            logger.info("LLM response received")

            assistant_message = response.choices[0].message
            logger.info(f"Tool calls: {assistant_message.tool_calls}, Content: {bool(assistant_message.content)}")

            # Handle tool calls
            if assistant_message.tool_calls:
                logger.info(f"Processing {len(assistant_message.tool_calls)} tool calls")
                messages.append(assistant_message.model_dump())

                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    agent_name = tool_name.replace("analyse_", "")

                    await self.send_json({
                        "type": "tool_call",
                        "tool": agent_name,
                    })

                    # Get result from cached analysis (instant!)
                    tool_result = self.execute_tool(agent_name)

                    await self.send_json({
                        "type": "tool_result",
                        "tool": agent_name,
                        "success": tool_result.get("success", False),
                    })

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(tool_result),
                    })

                # Stream final response
                await self.stream_response(model, messages)

            elif assistant_message.content:
                # Direct response without tools
                logger.info("Sending direct response (no tools)")
                await self.send_json({
                    "type": "token",
                    "text": assistant_message.content,
                })
                await self.send_json({
                    "type": "done",
                    "full_response": assistant_message.content,
                })
            else:
                logger.warning("No tool calls and no content in response")

        except Exception as e:
            logger.exception(f"Error processing question: {e}")
            await self.send_error(str(e))

    def execute_tool(self, agent_name: str) -> dict:
        """Execute a tool by querying cached analysis results."""
        if not self.track:
            return {"success": False, "error": "Track not found"}

        if self.track.status != Track.Status.READY:
            return {"success": False, "error": f"Track not ready: {self.track.status}"}

        if agent_name not in self.track.analysis:
            return {"success": False, "error": f"No {agent_name} analysis available"}

        data = self.track.analysis[agent_name]

        # Check if the analysis itself had an error
        if isinstance(data, dict) and "error" in data:
            return {"success": False, "error": data["error"]}

        return {"success": True, "data": data}

    async def stream_response(self, model: str, messages: list):
        """Stream the final LLM response token by token."""
        import litellm

        full_response = ""

        try:
            response = litellm.completion(
                model=model,
                messages=messages,
                stream=True,
            )

            for chunk in response:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_response += token
                    await self.send_json({
                        "type": "token",
                        "text": token,
                    })

            await self.send_json({
                "type": "done",
                "full_response": full_response,
            })

        except Exception as e:
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
