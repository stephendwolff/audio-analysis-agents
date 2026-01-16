# LLM Tool Integration - Educational Build Plan

## Overview

A phased approach to building LLM tool integration for the audio analysis agents project. Focus is on understanding MCP/LLM integration patterns, starting with raw SDK mechanics before introducing frameworks.

**Architecture Pattern:** Orchestrator as LLM client (Pattern A from DESIGN.md)

```
User question + track_id
        ↓
Orchestrator sends to LLM with tool schemas
        ↓
LLM returns tool call(s)
        ↓
Orchestrator runs appropriate agent(s)
        ↓
Results sent back to LLM
        ↓
LLM formulates natural language answer
        ↓
Response returned (streaming via WebSocket)
```

## Phases

### Phase 1: Gemini Function Calling Basics

**Goal:** Understand raw mechanics in isolation before touching the audio project.

**Build:**
- Standalone script with a simple tool (e.g., `get_weather(city)`)
- Manual request → function_call → response → final_answer loop

**Files:**
```
examples/01_gemini_basics.py
```

**Learn:**
- Gemini SDK setup and authentication
- Tool/function schema format
- The multi-turn tool calling loop
- When LLMs decide to call tools vs answer directly

---

### Phase 2: Synthetic Audio Eval Suite

**Goal:** Create deterministic test cases with known ground truth for tool selection accuracy.

**Build:**
- Generator for synthetic audio (sine waves, clicks, silence)
- Test manifest pairing questions → expected tool(s)

**Example test cases:**

| Audio | Question | Expected Tool(s) |
|-------|----------|------------------|
| 120 BPM click track | "What's the tempo?" | `rhythm` |
| 440Hz sine wave | "What frequency is this?" | `spectral` |
| Loud→quiet fade | "How does the volume change?" | `temporal` |
| Complex tone | "Describe this sound" | `spectral`, `temporal`, `rhythm` |

**Files:**
```
evals/
├── generate_synthetic.py   # Creates test audio files
├── fixtures/               # Generated .wav files
├── manifest.json           # Question → expected tools mapping
└── run_evals.py            # Runs eval suite, reports accuracy
```

**Learn:**
- Generating test audio with numpy/scipy
- Designing eval sets that isolate what you're testing
- Establishing baseline accuracy

**Future expansion (not built yet):**
- End-to-end answer quality scoring
- Real audio with human annotations
- Pre-computed analysis result mocks

---

### Phase 3: Wire Gemini to Audio Agents + Opik

**Goal:** Connect function calling to real agents with observability.

**Build:**
- Tool definitions matching existing agents (spectral, temporal, rhythm)
- LLM routing mode in orchestrator
- Opik tracing for all LLM calls and tool executions

**Files:**
```
src/orchestrator/
├── llm_routing.py          # Gemini client + tool definitions
└── orchestrator.py         # Add LLM routing mode

src/observability/
└── tracing.py              # Opik setup and decorators

examples/
└── 02_audio_tool_calling.py  # Interactive demo script
```

**Opik integration points:**
- Trace each LLM call (prompt, response, latency, tokens)
- Trace tool executions (which agent, duration, success/fail)
- Link traces to eval runs

**Learn:**
- Translating agent interfaces into LLM tool schemas
- Multi-turn conversation handling
- Setting up Opik observability
- Debugging tool selection via traces

**Eval checkpoint:** Run eval suite, measure tool selection accuracy.

---

### Phase 4: Refactor to LangGraph

**Goal:** See what a framework abstracts by rebuilding the same thing.

**Build:**
- LangGraph version of the orchestrator
- Config flag to switch between raw SDK and LangGraph

**Files:**
```
src/orchestrator/
├── llm_routing.py           # Keep raw Gemini version
├── llm_routing_langgraph.py # New LangGraph version
└── orchestrator.py          # Config flag to switch
```

**LangGraph concepts:**
- StateGraph - conversation flow as nodes/edges
- ToolNode - automatic tool execution
- Checkpointing - built-in state management
- Conditional edges - route based on LLM decisions

**Comparison:**

| Aspect | Raw SDK | LangGraph |
|--------|---------|-----------|
| Tool execution loop | You write it | Framework handles |
| Multi-tool calls | Manual orchestration | Automatic |
| Error handling | DIY | Configurable retry/fallback |
| State management | Your responsibility | Built-in |
| Debugging | Opik traces | Opik + LangSmith |
| Lines of code | ~80-100 | ~40-50 |
| Dependencies | 1 (google-generativeai) | 3+ (langgraph, langchain-core, etc) |

**Learn:**
- What LangGraph does under the hood (you just built it manually)
- Whether the abstraction is worth the dependency cost
- Comparing traces between implementations

**Eval checkpoint:** Both implementations should pass the same eval suite.

---

### Phase 5: Provider Swap (Optional)

**Goal:** Test portability by swapping Gemini for Claude.

**Build:**
- Abstract provider behind LiteLLM
- Run same eval suite on Claude

**Files:**
```
src/orchestrator/
└── llm_routing.py          # Abstract provider behind LiteLLM
```

**Learn:**
- Where provider differences bite (schema format, response structure)
- Whether LiteLLM abstraction is seamless or leaky
- Performance/cost comparison via Opik traces

---

### Phase 6: Example Frontend + Streaming API

**Goal:** Reference implementation showing full pattern, hostable anywhere.

**Constraints:**
- Static HTML/JS/CSS - no Django template variables
- Configurable API base URL (works from any host)
- API key auth (header-based)
- WebSocket streaming via Django Channels

**Build:**
- Minimal Django API layer
- WebSocket consumer for streaming LLM responses
- Static example frontend

**Files:**
```
example_frontend/
├── index.html              # Upload + chat UI
├── app.js                  # WebSocket handling, API calls
├── style.css               # Minimal styling
└── config.js               # API_BASE_URL, configurable

src/api/
├── views.py                # Upload endpoint
├── consumers.py            # WebSocket consumer (Channels)
├── auth.py                 # API key middleware
└── urls.py

config/
├── asgi.py                 # Updated for Channels
└── routing.py              # WebSocket routing
```

**User flow:**
1. Open index.html (served from anywhere)
2. Enter API URL + API key in config
3. Upload audio → POST /api/tracks/
4. Type question → WebSocket connects to /ws/chat/
5. LLM response streams back token-by-token
6. Answer builds up in real-time

**Dependencies:**
- `channels` (Django Channels)
- `gunicorn` + `uvicorn` workers (ASGI server)

**Learn:**
- Django Channels WebSocket setup
- Streaming LLM responses through WebSocket
- API key auth middleware pattern
- Decoupled frontend architecture

**Auth notes for production:**
- Current: API key header (X-API-Key)
- Future options: Session/cookie auth, JWT tokens

---

## Future Work (Not in Scope)

- **MCP HTTP Server (Pattern B)** - Expose tools to external LLM clients
- **End-to-end answer quality evals** - Score final answers, not just tool selection
- **Real audio test fixtures** - Annotated real-world audio samples
- **Session/JWT auth** - Production authentication patterns
- **Django models/persistence** - Full Track and AnalysisResult models

---

## Tech Stack Summary

**Core:**
- Python 3.10+
- Existing: librosa, numpy, soundfile, pydantic

**LLM Integration:**
- google-generativeai (Gemini SDK)
- langgraph, langchain-google-genai (Phase 4)
- litellm (Phase 5, optional)

**Observability:**
- opik

**API/Streaming:**
- Django + Django REST Framework
- channels (WebSocket support)
- gunicorn + uvicorn workers

---

## Project Structure (End State)

```
audio-analysis-agents/
├── config/
│   ├── settings.py
│   ├── urls.py
│   ├── asgi.py             # Channels routing
│   └── routing.py          # WebSocket routes
├── src/
│   ├── agents/             # Existing analysis agents
│   ├── orchestrator/
│   │   ├── orchestrator.py
│   │   ├── llm_routing.py           # Raw Gemini
│   │   └── llm_routing_langgraph.py # LangGraph version
│   ├── observability/
│   │   └── tracing.py      # Opik setup
│   ├── api/
│   │   ├── views.py
│   │   ├── consumers.py    # WebSocket
│   │   └── auth.py
│   └── tools/              # Existing audio loading
├── evals/
│   ├── generate_synthetic.py
│   ├── fixtures/
│   ├── manifest.json
│   └── run_evals.py
├── examples/
│   ├── 01_gemini_basics.py
│   └── 02_audio_tool_calling.py
├── example_frontend/
│   ├── index.html
│   ├── app.js
│   ├── style.css
│   └── config.js
├── tests/
└── docs/
    ├── DESIGN.md
    └── plans/
```
