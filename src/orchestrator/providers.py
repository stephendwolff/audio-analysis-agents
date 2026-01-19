"""
Analysis providers for the strategy pattern.

AnalysisProvider is an abstract base class defining how to fetch analysis data.
Implementations can query cached data from the database or run live analysis.

Note: Python's typing.Protocol offers an alternative approach using structural
typing (duck typing) rather than inheritance. With Protocol, any class with
matching method signatures is accepted, without needing to explicitly inherit.
"""

from abc import ABC, abstractmethod


class AnalysisProvider(ABC):
    """
    Abstract base class for analysis data providers.

    Subclasses implement different strategies for fetching analysis:
    - CachedAnalysisProvider: queries pre-computed results from database
    - LiveAnalysisProvider: runs agents on-demand
    - CacheFirstProvider: checks cache, falls back to live
    """

    @abstractmethod
    def get_analysis(self, agent_name: str, track_id: str) -> dict:
        """
        Get analysis result for an agent and track.

        Args:
            agent_name: Name of the agent (e.g., "rhythm", "spectral")
            track_id: UUID of the track

        Returns:
            Analysis data dictionary

        Raises:
            KeyError: If analysis is not available
            ValueError: If analysis contains an error
        """
        ...

    @abstractmethod
    def has_analysis(self, agent_name: str, track_id: str) -> bool:
        """
        Check if analysis is available.

        Args:
            agent_name: Name of the agent
            track_id: UUID of the track

        Returns:
            True if analysis exists and has no error
        """
        ...


class CachedAnalysisProvider(AnalysisProvider):
    """
    Provider that queries pre-computed analysis from the database.

    Uses Track.get_analysis() which was pre-computed by Celery workers.
    """

    def get_analysis(self, agent_name: str, track_id: str) -> dict:
        from src.api.models import Track
        track = Track.objects.get(id=track_id)
        return track.get_analysis(agent_name)

    def has_analysis(self, agent_name: str, track_id: str) -> bool:
        from src.api.models import Track
        try:
            track = Track.objects.get(id=track_id)
            return track.has_analysis(agent_name)
        except Track.DoesNotExist:
            return False
