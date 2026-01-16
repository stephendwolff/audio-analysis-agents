"""
WebSocket Consumer for Streaming Chat

Handles real-time chat with streaming LLM responses.

Protocol:
1. Client connects to /ws/chat/{track_id}/?token=<jwt> OR ?api_key=xxx
2. Client sends: {"type": "question", "text": "What's the tempo?"}
3. Server streams back:
   - {"type": "tool_call", "tool": "analyse_rhythm"}
   - {"type": "tool_result", "tool": "analyse_rhythm", "success": true}
   - {"type": "token", "text": "The"}
   - {"type": "token", "text": " tempo"}
   - {"type": "token", "text": " is"}
   - {"type": "done", "full_response": "The tempo is 120 BPM"}
"""

import json
import os
from urllib.parse import parse_qs

from channels.generic.websocket import AsyncWebsocketConsumer
from django.conf import settings
from rest_framework_simplejwt.exceptions import InvalidToken, TokenError
from rest_framework_simplejwt.tokens import AccessToken

from .auth import check_api_key
from .views import get_track_path


class ChatConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer for streaming audio analysis chat.

    Accepts either:
    - ?token=<jwt> - JWT authentication
    - ?api_key=<key> - Legacy API key authentication
    """

    async def connect(self):
        """Handle WebSocket connection."""
        self.track_id = self.scope["url_route"]["kwargs"]["track_id"]
        self.is_demo = False

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

        # Check track exists
        self.track_path = get_track_path(self.track_id)
        if not self.track_path:
            await self.close(code=4004)  # Not found
            return

        await self.accept()

    async def disconnect(self, close_code):
        """Handle WebSocket disconnection."""
        pass

    async def receive(self, text_data):
        """Handle incoming message from client."""
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

            await self.process_question(question)

        else:
            await self.send_error(f"Unknown message type: {msg_type}")

    async def process_question(self, question: str):
        """
        Process a question and stream the response.

        Uses LiteLLM for streaming support across providers.
        """
        import litellm

        # Import here to avoid circular imports
        from ..agents import SpectralAgent, TemporalAgent, RhythmAgent
        from ..tools.loader import load_audio

        # Tool definitions (same as LiteLLM router)
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

        # Agent instances
        agents = {
            "analyse_spectral": SpectralAgent(),
            "analyse_temporal": TemporalAgent(),
            "analyse_rhythm": RhythmAgent(),
        }

        # Build prompt
        prompt = f"""You have access to audio analysis tools. The user is asking about an audio file.

Audio file: {self.track_path}

User question: {question}

Use the appropriate analysis tool(s) to answer this question."""

        messages = [{"role": "user", "content": prompt}]
        model = getattr(settings, "LLM_MODEL", "gemini/gemini-2.0-flash")

        try:
            # First call - might return tool calls
            response = litellm.completion(
                model=model,
                messages=messages,
                tools=tools,
            )

            assistant_message = response.choices[0].message

            # Handle tool calls
            if assistant_message.tool_calls:
                messages.append(assistant_message.model_dump())

                # Load audio once
                audio = load_audio(self.track_path, target_sr=22050, mono=True)

                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name

                    await self.send_json({
                        "type": "tool_call",
                        "tool": tool_name.replace("analyse_", ""),
                    })

                    # Execute tool
                    if tool_name in agents:
                        agent = agents[tool_name]
                        result = agent.analyse(audio.samples, audio.sample_rate)
                        tool_result = {
                            "success": result.success,
                            "data": result.data,
                            "error": result.error,
                        }
                    else:
                        tool_result = {"error": f"Unknown tool: {tool_name}"}

                    await self.send_json({
                        "type": "tool_result",
                        "tool": tool_name.replace("analyse_", ""),
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
                await self.send_json({
                    "type": "token",
                    "text": assistant_message.content,
                })
                await self.send_json({
                    "type": "done",
                    "full_response": assistant_message.content,
                })

        except Exception as e:
            await self.send_error(str(e))

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
