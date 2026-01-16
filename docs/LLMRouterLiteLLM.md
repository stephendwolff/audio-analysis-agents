# LLMRouterLiteLLM

Provider-agnostic LLM router using [LiteLLM](https://docs.litellm.ai/). The same code works with Claude, Gemini, GPT-4, and 100+ other models.

## Usage

```python
from src.orchestrator import LLMRouterLiteLLM

# Use Claude
router = LLMRouterLiteLLM(model="claude-sonnet-4-20250514")

# Use Gemini
router = LLMRouterLiteLLM(model="gemini/gemini-2.0-flash")

# Use GPT-4
router = LLMRouterLiteLLM(model="gpt-4o")

result = router.process_question(
    question="What's the tempo?",
    audio_path="track.wav"
)
```

## Environment Variables

Set the API key for your chosen provider:

| Provider | Environment Variable |
|----------|---------------------|
| Anthropic (Claude) | `ANTHROPIC_API_KEY` |
| Google (Gemini) | `GOOGLE_API_KEY` |
| OpenAI (GPT-4) | `OPENAI_API_KEY` |

## Model Identifiers

LiteLLM uses specific prefixes for some providers:

| Provider | Model Format | Example |
|----------|--------------|---------|
| Anthropic | `claude-*` | `claude-sonnet-4-20250514` |
| Google | `gemini/*` | `gemini/gemini-2.0-flash` |
| OpenAI | `gpt-*` | `gpt-4o` |

See [LiteLLM supported models](https://docs.litellm.ai/docs/providers) for the full list.

## How It Works

LiteLLM translates OpenAI's tool calling format to each provider's native format:

```
Your code (OpenAI format)
         │
         ▼
    ┌─────────┐
    │ LiteLLM │
    └────┬────┘
         │ translates to native format
    ┌────┴────┬────────────┐
    ▼         ▼            ▼
 Claude    Gemini       GPT-4
```

### Tool Definition Format

LiteLLM uses OpenAI's tool format:

```python
{
    "type": "function",
    "function": {
        "name": "analyse_rhythm",
        "description": "Analyse tempo and rhythmic properties...",
        "parameters": {
            "type": "object",
            "properties": {
                "track_id": {"type": "string", "description": "..."}
            },
            "required": ["track_id"]
        }
    }
}
```

### Message Format

Messages use OpenAI's format:

```python
messages = [
    {"role": "user", "content": "What's the tempo?"},
    {"role": "assistant", "tool_calls": [...]},  # LLM wants to call tools
    {"role": "tool", "tool_call_id": "...", "content": "..."},  # Tool result
    {"role": "assistant", "content": "The tempo is 120 BPM"}  # Final answer
]
```

## Comparison with Other Implementations

| Aspect | Raw SDK | LangGraph | LiteLLM |
|--------|---------|-----------|---------|
| Provider lock-in | Gemini only | Gemini only | Any provider |
| Tool format | Google native | LangChain | OpenAI (translated) |
| Dependencies | 1 | 3+ | 1 |
| Switching providers | Rewrite code | Rewrite code | Change model string |

## Eval Results

All providers achieve 100% tool selection accuracy:

| Provider | Model | Accuracy |
|----------|-------|----------|
| Google | gemini-2.0-flash | 14/14 |
| Anthropic | claude-sonnet-4-20250514 | 14/14 |

## Running Evals

```bash
# Test with Claude
python evals/run_evals.py --litellm --model claude-sonnet-4-20250514

# Test with Gemini
python evals/run_evals.py --litellm --model gemini/gemini-2.0-flash

# Test with GPT-4
python evals/run_evals.py --litellm --model gpt-4o
```

## When to Use LiteLLM

**Use LiteLLM when:**
- You want to compare models from different providers
- You need to switch providers without code changes
- You're building a product that should support multiple LLMs
- You want a unified API across providers

**Use provider-specific SDK when:**
- You need provider-specific features (e.g., Claude's extended thinking)
- You want minimal dependencies
- You need the lowest possible latency
- You're deeply integrated with one provider's ecosystem

## File Location

`src/orchestrator/llm_routing_litellm.py`
