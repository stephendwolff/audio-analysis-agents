"""Main orchestrator for coordinating analysis agents."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..agents import (
    BaseAgent,
    AnalysisResult,
    SpectralAgent,
    TemporalAgent,
    RhythmAgent,
)
from ..tools.loader import load_audio, AudioData


class AnalysisReport(BaseModel):
    """Complete analysis report from orchestrator."""

    file_path: str
    duration_seconds: float
    sample_rate: int
    analyses: dict[str, AnalysisResult]
    summary: dict[str, Any] = {}

    class Config:
        arbitrary_types_allowed = True


class Orchestrator:
    """
    Coordinates analysis agents to process audio files.

    The orchestrator:
    - Loads audio files
    - Dispatches to appropriate agents
    - Aggregates and returns results
    """

    def __init__(self):
        """Initialise with default agents."""
        self.agents: dict[str, BaseAgent] = {}
        self._register_default_agents()

    def _register_default_agents(self):
        """Register the built-in agents."""
        self.register_agent(SpectralAgent())
        self.register_agent(TemporalAgent())
        self.register_agent(RhythmAgent())

    def register_agent(self, agent: BaseAgent):
        """
        Register an analysis agent.

        Args:
            agent: Agent instance to register
        """
        self.agents[agent.name] = agent

    def list_agents(self) -> list[str]:
        """List available agent names."""
        return list(self.agents.keys())

    def analyse(
        self,
        file_path: str | Path,
        tasks: list[str] | None = None,
        target_sr: int | None = 22050,
    ) -> AnalysisReport:
        """
        Analyse an audio file with specified agents.

        Args:
            file_path: Path to audio file
            tasks: List of agent names to run (None for all)
            target_sr: Target sample rate for analysis

        Returns:
            AnalysisReport with results from all agents
        """
        # Load audio
        audio = load_audio(file_path, target_sr=target_sr, mono=True)

        # Determine which agents to run
        agents_to_run = tasks or list(self.agents.keys())

        # Run analyses
        results: dict[str, AnalysisResult] = {}
        for agent_name in agents_to_run:
            if agent_name not in self.agents:
                results[agent_name] = AnalysisResult(
                    agent=agent_name,
                    success=False,
                    error=f"Unknown agent: {agent_name}",
                )
                continue

            agent = self.agents[agent_name]
            results[agent_name] = agent.analyse(audio.samples, audio.sample_rate)

        # Build report
        report = AnalysisReport(
            file_path=str(file_path),
            duration_seconds=audio.duration,
            sample_rate=audio.sample_rate,
            analyses=results,
            summary=self._build_summary(results),
        )

        return report

    def _build_summary(self, results: dict[str, AnalysisResult]) -> dict[str, Any]:
        """Build a high-level summary from analysis results."""
        summary = {}

        # Extract key metrics if available
        if "temporal" in results and results["temporal"].success:
            temporal = results["temporal"].data
            summary["duration"] = temporal.get("duration_seconds")
            summary["dynamic_range_db"] = temporal.get("dynamic_range_db")

        if "rhythm" in results and results["rhythm"].success:
            rhythm = results["rhythm"].data
            summary["tempo_bpm"] = rhythm.get("tempo_bpm")
            summary["beat_count"] = rhythm.get("beat_count")

        if "spectral" in results and results["spectral"].success:
            spectral = results["spectral"].data
            if "spectral_centroid" in spectral:
                summary["brightness"] = spectral["spectral_centroid"].get("mean")
            if "spectral_flatness" in spectral:
                summary["tonality"] = spectral["spectral_flatness"].get("interpretation")

        return summary

    def analyse_batch(
        self,
        file_paths: list[str | Path],
        tasks: list[str] | None = None,
    ) -> list[AnalysisReport]:
        """
        Analyse multiple audio files.

        Args:
            file_paths: List of paths to audio files
            tasks: List of agent names to run (None for all)

        Returns:
            List of AnalysisReports
        """
        return [self.analyse(fp, tasks=tasks) for fp in file_paths]
