# LLMRouterLangGraph

The LangGraph implementation of the LLM router. Provides the same functionality as `LLMRouter` but uses LangGraph's StateGraph abstraction.

## Usage

```python
from src.orchestrator import LLMRouterLangGraph

router = LLMRouterLangGraph()
result = router.process_question(
    question="What's the tempo?",
    audio_path="path/to/audio.wav"
)
```

The interface is identical to `LLMRouter`.

## Key Differences from Raw SDK

| Aspect | Raw SDK (LLMRouter) | LangGraph (LLMRouterLangGraph) |
|--------|---------------------|--------------------------------|
| Conversation loop | Manual `while True` loop | Graph traversal |
| Tool execution | Manual iteration over function_calls | `ToolNode` handles automatically |
| State management | Manual `contents` list | `AgentState` TypedDict |
| Conditional routing | `if function_calls:` | `add_conditional_edges()` |
| Lines of code | ~170 | ~200 (more structure, less logic) |
| Dependencies | 1 (google-genai) | 3 (langgraph, langchain-core, langchain-google-genai) |

## LangGraph Concepts

### StateGraph

The conversation flow is defined as a graph of nodes and edges:

```
    ┌─────────────┐
    │   START     │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │ call_model  │◄─────────────┐
    └──────┬──────┘              │
           │                     │
           ▼                     │
    ┌─────────────┐              │
    │should_continue│            │
    └──────┬──────┘              │
           │                     │
   ┌───────┴───────┐             │
   │               │             │
   ▼               ▼             │
┌─────┐      ┌──────────┐        │
│ END │      │  tools   │────────┘
└─────┘      │(ToolNode)│
             └──────────┘
```

### State (AgentState)

A TypedDict that flows through the graph:

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # Conversation history
    audio_path: str                           # Current audio file
    tools_called: list[str]                   # Tracking
    analyses: dict[str, Any]                  # Results
```

The `add_messages` annotation tells LangGraph to append new messages rather than replace.

### Nodes

Functions that process state and return updates:

```python
def call_model(state: AgentState) -> dict:
    """Call the LLM with current messages."""
    model = ChatGoogleGenerativeAI(...).bind_tools(TOOLS)
    response = model.invoke(state["messages"])
    return {"messages": [response]}  # Appended to state
```

### ToolNode

LangGraph's built-in node that:
1. Extracts tool calls from the last AI message
2. Executes each tool
3. Returns ToolMessage results

```python
graph.add_node("tools", ToolNode(tools=TOOLS))
```

This replaces ~30 lines of manual tool execution code.

### Conditional Edges

Route based on LLM output:

```python
def should_continue(state: AgentState) -> Literal["tools", "end"]:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"

graph.add_conditional_edges("call_model", should_continue, {
    "tools": "tools",
    "end": END,
})
```

### Tools

Defined using `@tool` decorator. The docstring becomes the tool description:

```python
@tool
def analyse_rhythm(track_id: str) -> dict:
    """
    Analyse tempo and rhythmic properties of audio. Returns estimated BPM,
    beat positions, onset times, and tempo stability.
    Use for questions about tempo, BPM, beats, rhythm, or timing.
    """
    ...
```

## What LangGraph Handles For You

1. **Message accumulation** - The `add_messages` annotation handles appending
2. **Tool execution loop** - `ToolNode` iterates over all tool calls
3. **Tool response formatting** - Converts results to `ToolMessage`
4. **Graph traversal** - Manages the back-and-forth until completion

## What You Still Handle

1. **Tool implementations** - Your actual analysis code
2. **State schema design** - What data flows through the graph
3. **Routing logic** - When to call tools vs end
4. **Prompt engineering** - Initial message construction

## Trade-offs

**Advantages:**
- Less boilerplate for the conversation loop
- Built-in patterns for common flows
- LangSmith integration for debugging
- Easier to add complexity (retries, human-in-the-loop, checkpoints)

**Disadvantages:**
- More dependencies (langchain ecosystem)
- Abstraction can hide what's happening
- Slightly more verbose for simple cases
- Learning curve for LangGraph concepts

## Eval Results

Both implementations achieve similar tool selection accuracy:

| Implementation | Accuracy |
|----------------|----------|
| Raw SDK | 14/14 (100%) |
| LangGraph | 13/14 (92.9%) |

The difference is on ambiguous questions where LLM behavior can vary.

## When to Use Which

**Use raw SDK when:**
- You want minimal dependencies
- The flow is simple (single tool call pattern)
- You need maximum control
- You're learning how tool calling works

**Use LangGraph when:**
- You need complex flows (parallel tool calls, human-in-the-loop)
- You want built-in checkpointing/state persistence
- You're using other LangChain components
- You want LangSmith observability

## File Location

`src/orchestrator/llm_routing_langgraph.py`
