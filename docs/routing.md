# LLM Routing

The orchestrator connects natural language questions to analysis agents via LLM tool calling. Three implementations exist, each using a different integration approach.

## How It Works

```
User: "What's the tempo and frequency content?"
         │
         ▼
┌──────────────────────────────────────────────┐
│  LLM sees tool descriptions, picks agents    │
│  → analyse_rhythm, analyse_spectral          │
└──────────────────────────────────────────────┘
         │
         ▼ Execute tools, return results to LLM
         │
         ▼
LLM: "The tempo is 120 BPM and the dominant frequency is 440 Hz"
```

The LLM never sees Python code. It sees JSON schemas describing available tools, decides which to call based on the question, and formulates a natural language answer from the results.

## Usage

All three implementations share the same interface:

```python
router = SomeRouter()
result = router.process_question(
    question="What's the tempo of this track?",
    audio_path="path/to/audio.wav"
)

print(result["response"])      # "The tempo is approximately 120 BPM"
print(result["tools_called"])  # ["analyse_rhythm"]
print(result["analyses"])      # {"analyse_rhythm": {"success": True, "data": {...}}}
```

## Implementations

### LLMRouterLiteLLM (recommended)

Provider-agnostic via [LiteLLM](https://docs.litellm.ai/). The same code works with Claude, Gemini, GPT-4, and 100+ other models.

```python
from src.orchestrator import LLMRouterLiteLLM

router = LLMRouterLiteLLM(model="gemini/gemini-2.0-flash")  # or claude-sonnet-4-20250514, gpt-4o
```

Uses OpenAI's tool format internally. LiteLLM translates to each provider's native format.

**File:** `src/orchestrator/llm_routing_litellm.py`
**Deps:** `pip install -e ".[litellm]"`

| Provider | Env Variable | Model Format |
|----------|-------------|--------------|
| Google | `GOOGLE_API_KEY` | `gemini/gemini-2.0-flash` |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-sonnet-4-20250514` |
| OpenAI | `OPENAI_API_KEY` | `gpt-4o` |

### LLMRouter (Gemini SDK)

Direct Gemini integration. Fewer dependencies, Gemini-only.

```python
from src.orchestrator import LLMRouter

router = LLMRouter(model="gemini-2.0-flash")
```

Manages the conversation loop manually: send question, receive tool calls, execute tools, send results back, receive final answer.

**File:** `src/orchestrator/llm_routing.py`
**Deps:** `pip install -e ".[llm]"`

### LLMRouterLangGraph

LangGraph state machine abstraction. Same Gemini backend, but the conversation loop is expressed as a graph.

```python
from src.orchestrator import LLMRouterLangGraph

router = LLMRouterLangGraph()
```

Replaces the manual conversation loop with a `StateGraph`:

```
START → call_model → should_continue? → tools → call_model → ... → END
```

LangGraph handles message accumulation, tool execution, and routing. You handle tool implementations and state schema.

**File:** `src/orchestrator/llm_routing_langgraph.py`
**Deps:** `pip install -e ".[langgraph]"`

## Comparison

| | LiteLLM | Raw SDK | LangGraph |
|---|---------|---------|-----------|
| **Providers** | Any (100+) | Gemini only | Gemini only |
| **Switching providers** | Change model string | Rewrite code | Rewrite code |
| **Dependencies** | 1 (litellm) | 1 (google-genai) | 3+ (langgraph, langchain) |
| **Conversation loop** | Manual while-loop | Manual while-loop | Graph traversal |
| **Tool format** | OpenAI (translated) | Google native | LangChain @tool |
| **Eval accuracy** | 14/14 (100%) | 14/14 (100%) | 13/14 (92.9%) |

## When to Use Which

**LiteLLM** -- default choice. Multi-provider, minimal dependencies, easy to switch models.

**Raw Gemini SDK** -- when you need Gemini-specific features or want to understand the conversation loop mechanics.

**LangGraph** -- when you need complex flows (parallel tool calls, human-in-the-loop, checkpoints) or are already in the LangChain ecosystem.

## Evaluation

The `evals/` directory contains a tool selection evaluation suite:

```bash
# Test with different providers
python evals/run_evals.py --litellm --model gemini/gemini-2.0-flash
python evals/run_evals.py --litellm --model claude-sonnet-4-20250514
python evals/run_evals.py --litellm --model gpt-4o
```

## Tracing

All routers support Opik tracing. Set `OPIK_API_KEY` to enable:

- LLM calls: prompt, response, model, latency
- Tool executions: agent name, duration, success/failure

## Tool Selection

Tools are auto-generated from the agent registry. Each agent's `description` field becomes the tool description the LLM uses to decide which agent to invoke. Write descriptions as if explaining the tool to an LLM -- include hints about when to use each one.

```python
from src.agents import get_tool_schemas_openai

schemas = get_tool_schemas_openai()
# [{"type": "function", "function": {"name": "analyse_rhythm", "description": "...", ...}}, ...]
```
