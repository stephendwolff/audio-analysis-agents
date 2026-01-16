"""
Tool Selection Eval Runner

Tests whether the LLM selects the correct analysis tools for different questions.
This evaluates TOOL SELECTION accuracy, not the actual analysis results.

Run with: python evals/run_evals.py
Options:
  --verbose    Show details for each test case
  --category   Only run tests in a specific category
  --id         Run a single test by ID
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Add parent to path for src imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.orchestrator import LLMRouter

load_dotenv()


# ---------------------------------------------------------------------------
# Eval Runner
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Result of a single eval test case."""
    test_id: str
    question: str
    expected_tools: list[str]
    actual_tools: list[str]
    passed: bool
    error: str | None = None


def run_eval(test_case: dict, router: LLMRouter, verbose: bool = False) -> EvalResult:
    """Run a single eval test case."""
    test_id = test_case["id"]
    question = test_case["question"]
    expected = set(test_case["expected_tools"])
    audio_file = test_case["audio_file"]

    # Use the audio filename as track_id for context
    track_id = Path(audio_file).stem

    try:
        # Use the router's get_tool_selection - same code path as production
        actual = router.get_tool_selection(question, track_id)
        actual_set = set(actual)

        # Check if selection matches expected
        passed = actual_set == expected

        result = EvalResult(
            test_id=test_id,
            question=question,
            expected_tools=list(expected),
            actual_tools=actual,
            passed=passed
        )

    except Exception as e:
        result = EvalResult(
            test_id=test_id,
            question=question,
            expected_tools=list(expected),
            actual_tools=[],
            passed=False,
            error=str(e)
        )

    if verbose:
        status = "PASS" if result.passed else "FAIL"
        print(f"\n[{status}] {test_id}")
        print(f"  Question: {question}")
        print(f"  Expected: {result.expected_tools}")
        print(f"  Actual:   {result.actual_tools}")
        if result.error:
            print(f"  Error:    {result.error}")

    return result


def load_manifest(manifest_path: Path) -> dict:
    """Load the eval manifest."""
    with open(manifest_path) as f:
        return json.load(f)


def run_all_evals(
    manifest_path: Path,
    verbose: bool = False,
    category: str | None = None,
    test_id: str | None = None
) -> list[EvalResult]:
    """Run all eval test cases."""
    manifest = load_manifest(manifest_path)
    test_cases = manifest["test_cases"]

    # Filter by category if specified
    if category:
        test_cases = [tc for tc in test_cases if tc.get("category") == category]
        print(f"Running {len(test_cases)} tests in category '{category}'")

    # Filter by ID if specified
    if test_id:
        test_cases = [tc for tc in test_cases if tc["id"] == test_id]
        print(f"Running single test: {test_id}")

    if not test_cases:
        print("No test cases found!")
        return []

    # Create router (disable tracing for evals to reduce noise)
    router = LLMRouter(enable_tracing=False)

    results = []
    for i, test_case in enumerate(test_cases, 1):
        if not verbose:
            print(f"Running test {i}/{len(test_cases)}: {test_case['id']}", end="\r")
        result = run_eval(test_case, router, verbose=verbose)
        results.append(result)

    return results


def print_summary(results: list[EvalResult]):
    """Print eval summary."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed

    print("\n" + "=" * 60)
    print("EVAL SUMMARY")
    print("=" * 60)
    print(f"Total:  {total}")
    print(f"Passed: {passed} ({100*passed/total:.1f}%)")
    print(f"Failed: {failed}")

    if failed > 0:
        print("\nFailed tests:")
        for r in results:
            if not r.passed:
                print(f"  - {r.test_id}: expected {r.expected_tools}, got {r.actual_tools}")
                if r.error:
                    print(f"    Error: {r.error}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run tool selection evals")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show details for each test")
    parser.add_argument("--category", "-c", help="Only run tests in this category")
    parser.add_argument("--id", help="Run a single test by ID")
    args = parser.parse_args()

    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not set")
        sys.exit(1)

    manifest_path = Path(__file__).parent / "manifest.json"
    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}")
        sys.exit(1)

    print("=" * 60)
    print("TOOL SELECTION EVAL")
    print("=" * 60)

    results = run_all_evals(
        manifest_path,
        verbose=args.verbose,
        category=args.category,
        test_id=args.id
    )

    print_summary(results)

    # Exit with error code if any failed
    sys.exit(0 if all(r.passed for r in results) else 1)


if __name__ == "__main__":
    main()
