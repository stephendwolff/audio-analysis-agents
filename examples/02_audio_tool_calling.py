"""
Phase 3: Audio Tool Calling with Real Agents

This script demonstrates the full LLM → Agent pipeline:
1. Load a real audio file
2. Ask questions via Gemini
3. Gemini calls the appropriate analysis agents
4. Agents return real analysis results
5. Gemini formulates a natural language answer

Run with: python examples/02_audio_tool_calling.py [audio_file]

If no audio file is provided, uses a synthetic test fixture.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.orchestrator import LLMRouter


def main():
    # Determine audio file to use
    if len(sys.argv) > 1:
        audio_path = Path(sys.argv[1])
        if not audio_path.exists():
            print(f"Error: File not found: {audio_path}")
            sys.exit(1)
    else:
        # Use a synthetic fixture
        audio_path = Path(__file__).parent.parent / "evals" / "fixtures" / "complex.wav"
        if not audio_path.exists():
            print("No synthetic fixtures found. Run: python evals/generate_synthetic.py")
            sys.exit(1)
        print(f"Using synthetic audio: {audio_path.name}")

    print("\n" + "=" * 60)
    print("AUDIO ANALYSIS WITH LLM ROUTING")
    print("=" * 60)
    print(f"Audio file: {audio_path}")
    print("=" * 60)

    # Initialise the router (this also initialises Opik tracing)
    router = LLMRouter(enable_tracing=True)

    # Demo questions
    questions = [
        "What's the tempo of this audio?",
        "What are the dominant frequencies?",
        "Describe this audio - tempo, frequencies, and dynamics.",
    ]

    for question in questions:
        print(f"\n{'─'*60}")
        result = router.process_question(question, audio_path, verbose=True)

        # Show which tools were called
        print(f"\nTools called: {result['tools_called']}")

    print("\n" + "=" * 60)
    print("Check your Opik dashboard for traces!")
    print("=" * 60)


def interactive():
    """Run in interactive mode - ask your own questions."""
    if len(sys.argv) < 2:
        print("Usage: python examples/02_audio_tool_calling.py <audio_file> --interactive")
        sys.exit(1)

    audio_path = Path(sys.argv[1])
    if not audio_path.exists():
        print(f"Error: File not found: {audio_path}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("INTERACTIVE AUDIO ANALYSIS")
    print("=" * 60)
    print(f"Audio file: {audio_path}")
    print("Type 'quit' to exit")
    print("=" * 60)

    router = LLMRouter(enable_tracing=True)

    while True:
        print()
        question = input("Question: ").strip()

        if question.lower() in ("quit", "exit", "q"):
            break

        if not question:
            continue

        result = router.process_question(question, audio_path, verbose=True)
        print(f"\nTools called: {result['tools_called']}")


if __name__ == "__main__":
    if "--interactive" in sys.argv:
        sys.argv.remove("--interactive")
        interactive()
    else:
        main()
