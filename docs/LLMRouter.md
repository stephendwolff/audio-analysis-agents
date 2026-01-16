# LLMRouter

The `LLMRouter` connects Gemini to the audio analysis agents. It handles the multi-turn conversation loop where:

1. User asks a question
2. Gemini decides which tool(s) to call
3. Tools execute and return results
4. Gemini formulates a natural language answer

## Usage

```python
from src.orchestrator import LLMRouter

router = LLMRouter()
result = router.process_question(
    question="What's the tempo of this track?",
    audio_path="path/to/audio.wav"
)

print(result["response"])      # "The tempo is approximately 120 BPM"
print(result["tools_called"])  # ["analyse_rhythm"]
print(result["analyses"])      # {"analyse_rhythm": {"success": True, "data": {...}}}
```

## How It Works

### Tool Definitions

The LLM never sees Python code. It only sees JSON schemas describing what tools are available:

```python
types.FunctionDeclaration(
    name="analyse_rhythm",
    description="Analyse tempo and rhythmic properties of audio. Returns estimated BPM, "
                "beat positions, onset times, and tempo stability. "
                "Use for questions about tempo, BPM, beats, rhythm, or timing.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "track_id": types.Schema(type=types.Type.STRING, description="...")
        },
        required=["track_id"]
    )
)
```

The `description` field is critical - it's how Gemini decides which tool to use. Include hints about when to use each tool.

### The Conversation Loop

```
User: "What's the tempo?"
         │
         ▼
┌─────────────────────────────────────────┐
│  contents = [user message]              │
│  response = generate_content(contents)  │
└─────────────────────────────────────────┘
         │
         ▼ Gemini returns: function_call("analyse_rhythm", {track_id: "..."})
         │
┌─────────────────────────────────────────┐
│  contents += [assistant's func call]    │
│  result = execute_tool("rhythm", ...)   │
│  contents += [function response]        │
│  response = generate_content(contents)  │◄── Loop back
└─────────────────────────────────────────┘
         │
         ▼ Gemini returns: text("The tempo is 120 BPM")
         │
    Return response
```

**Key insight:** The `contents` list is the conversation history. Each iteration:
1. Send history to Gemini
2. Gemini returns either function calls OR text
3. If function calls: execute them, add results to history, loop
4. If text: we're done, return

### Conversation History Structure

After one tool call, `contents` looks like:

```python
[
    # 1. Original user message
    Content(role="user", parts=[Part(text="You have access to... What's the tempo?")]),

    # 2. Gemini's response requesting a tool call
    Content(role="model", parts=[Part(function_call={"name": "analyse_rhythm", "args": {...}})]),

    # 3. Tool execution result (sent as "user" role - Gemini's convention)
    Content(role="user", parts=[Part(function_response={"name": "analyse_rhythm", "response": {...}})]),
]
```

Then we call `generate_content(contents)` again, and Gemini sees the full history including the tool result.

### Multiple Tool Calls

Gemini can request multiple tools in a single response:

```
User: "What's the tempo and frequency content?"
         │
         ▼
    Gemini returns: [function_call("analyse_rhythm"), function_call("analyse_spectral")]
         │
         ▼
    Execute both, add both results to history
         │
         ▼
    Gemini returns: text("The tempo is 120 BPM and the dominant frequency is 440 Hz")
```

The code handles this by iterating over all function calls:

```python
for func_call in function_calls:
    result = self._execute_tool(func_name, track_id)
    function_responses.append(Part.from_function_response(...))

contents.append(Content(role="user", parts=function_responses))
```

## Components

### `__init__(model, enable_tracing, project_name)`

- Creates Gemini client
- Initialises Opik tracing (if enabled)
- Pre-instantiates all agents (they're stateless)
- Sets up audio cache

### `_load_audio(audio_path)`

Loads audio file using `src/tools/loader.py`. Caches results so repeated tool calls on the same file don't reload from disk.

### `_execute_tool(tool_name, track_id)`

Runs an analysis agent:

```python
agent = self._agents[tool_name]           # Get pre-instantiated agent
audio = self._load_audio(track_id)        # Load (or get cached) audio
result = agent.analyse(audio.samples, audio.sample_rate)  # Run analysis
return {"success": True, "data": result.data}
```

Wrapped in `trace_tool_execution()` for Opik observability.

### `process_question(question, audio_path, verbose)`

Main entry point. Returns:

```python
{
    "response": "The tempo is 120 BPM",           # Final text answer
    "tools_called": ["analyse_rhythm"],           # Which tools were invoked
    "analyses": {"analyse_rhythm": {"success": True, "data": {...}}}  # Raw results
}
```

### `get_tool_selection(question, track_id)`

For evaluation only. Returns which tools Gemini would select without executing them:

```python
router.get_tool_selection("What's the BPM?", "track123")
# Returns: ["rhythm"]
```

## Tracing

When `enable_tracing=True`, the router sends traces to Opik:

- **LLM calls**: prompt, response, model, latency
- **Tool executions**: agent name, track_id, duration, success/failure

View traces at your Opik dashboard.

## Error Handling

Tool execution errors are caught and returned to Gemini:

```python
{"success": False, "error": "Audio file not found: missing.wav"}
```

Gemini then incorporates the error into its response, e.g., *"I couldn't analyse that file because it wasn't found."*

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `"gemini-2.0-flash"` | Gemini model to use |
| `enable_tracing` | `True` | Send traces to Opik |
| `project_name` | `"audio-analysis-agents"` | Opik project name |

## File Location

`src/orchestrator/llm_routing.py`
