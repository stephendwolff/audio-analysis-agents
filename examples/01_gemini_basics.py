"""
Phase 1: Gemini Function Calling Basics

This script demonstrates the core mechanics of LLM tool calling:
1. Define a tool schema (what the LLM can call)
2. Send a prompt with the tool definition
3. Handle the LLM's function call response
4. Execute the function and send the result back
5. Get the final natural language answer

Run with: python examples/01_gemini_basics.py
"""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load API key from .env file
load_dotenv()

# ---------------------------------------------------------------------------
# SECTION 1: Tool Definition
# ---------------------------------------------------------------------------
# This schema tells Gemini what functions it can call. The LLM never executes
# code - it just returns a structured request saying "call this function with
# these arguments". You execute it and send back the result.

get_audio_duration_declaration = types.FunctionDeclaration(
    name="get_audio_duration",
    description="Get the duration of an audio file in seconds",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "file_path": types.Schema(
                type=types.Type.STRING,
                description="Path to the audio file"
            )
        },
        required=["file_path"]
    )
)

get_audio_tempo_declaration = types.FunctionDeclaration(
    name="get_audio_tempo",
    description="Detect the tempo (BPM) of an audio file",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "file_path": types.Schema(
                type=types.Type.STRING,
                description="Path to the audio file"
            )
        },
        required=["file_path"]
    )
)

# Bundle tools together
audio_tools = types.Tool(
    function_declarations=[get_audio_duration_declaration, get_audio_tempo_declaration]
)


# ---------------------------------------------------------------------------
# SECTION 2: Tool Implementation
# ---------------------------------------------------------------------------
# These are the actual functions that get called. In the real project, these
# would call the analysis agents. Here we use mock data.

def get_audio_duration(file_path: str) -> dict:
    """Mock implementation - returns fake duration."""
    print(f"  [TOOL] get_audio_duration called with: {file_path}")
    # In reality: load file with librosa, return actual duration
    return {"duration_seconds": 180.5, "file": file_path}


def get_audio_tempo(file_path: str) -> dict:
    """Mock implementation - returns fake tempo."""
    print(f"  [TOOL] get_audio_tempo called with: {file_path}")
    # In reality: run rhythm agent, return actual BPM
    return {"tempo_bpm": 120, "confidence": 0.95, "file": file_path}


# Map function names to implementations
TOOL_FUNCTIONS = {
    "get_audio_duration": get_audio_duration,
    "get_audio_tempo": get_audio_tempo,
}


# ---------------------------------------------------------------------------
# SECTION 3: Conversation Loop
# ---------------------------------------------------------------------------
# This is the core pattern: send message → check for function calls →
# execute and send results → repeat until LLM gives final answer.

def chat_with_tools(user_message: str) -> str:
    """
    Send a message to Gemini with tools available.

    Returns the final text response after handling any tool calls.
    """
    # Create the client
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    print(f"\n{'='*60}")
    print(f"USER: {user_message}")
    print(f"{'='*60}")

    # Build initial conversation history
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)]
        )
    ]

    # Loop until we get a final text response
    while True:
        # Send message with tools
        response = client.models.generate_content(
            model="gemini-2.0-flash",  # Current fast model
            contents=contents,
            config=types.GenerateContentConfig(
                tools=[audio_tools]
            )
        )

        # Get the response parts
        candidate = response.candidates[0]
        parts = candidate.content.parts

        # Check what kind of response we got
        function_calls = []
        text_response = None

        for part in parts:
            if part.function_call:
                function_calls.append(part.function_call)
            elif part.text:
                text_response = part.text

        # If there are function calls, execute them
        if function_calls:
            # Add the assistant's response to history
            contents.append(candidate.content)

            # Process each function call
            function_responses = []
            for func_call in function_calls:
                func_name = func_call.name
                func_args = dict(func_call.args)

                print(f"\n[LLM wants to call: {func_name}({func_args})]")

                # Execute the function
                if func_name in TOOL_FUNCTIONS:
                    result = TOOL_FUNCTIONS[func_name](**func_args)
                    print(f"  [TOOL returned: {result}]")

                    function_responses.append(
                        types.Part.from_function_response(
                            name=func_name,
                            response={"result": result}
                        )
                    )
                else:
                    print(f"  [ERROR: Unknown function {func_name}]")
                    function_responses.append(
                        types.Part.from_function_response(
                            name=func_name,
                            response={"error": f"Unknown function: {func_name}"}
                        )
                    )

            # Add function responses to history and continue
            contents.append(
                types.Content(
                    role="user",
                    parts=function_responses
                )
            )
            # Loop continues - send function results back to LLM

        elif text_response:
            # Final text response - we're done
            print(f"\n{'='*60}")
            print(f"ASSISTANT: {text_response}")
            print(f"{'='*60}")
            return text_response

        else:
            print("[WARNING: No function calls or text in response]")
            return "No response generated"


# ---------------------------------------------------------------------------
# SECTION 4: Main - Try it out
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not set")
        print("Create a .env file with: GOOGLE_API_KEY=your-key-here")
        exit(1)

    # Try some questions - notice how different questions trigger different tools
    questions = [
        "How long is the audio file at /music/song.mp3?",
        "What's the BPM of /music/beat.wav?",
        "Tell me about /music/track.mp3 - how long is it and what's the tempo?",
    ]

    print("\n" + "="*60)
    print("GEMINI FUNCTION CALLING DEMO")
    print("="*60)

    for question in questions:
        chat_with_tools(question)
        print("\n")
