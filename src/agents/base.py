"""Base agent class and common types."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from pydantic import BaseModel


class AnalysisResult(BaseModel):
    """Result from an analysis agent."""

    agent: str
    success: bool
    data: dict[str, Any] = {}
    error: str | None = None

    class Config:
        arbitrary_types_allowed = True


class BaseAgent(ABC):
    """Base class for analysis agents."""

    name: str
    description: str

    def __init__(self):
        # Validate required class attributes
        if not hasattr(self, 'name') or not self.name:
            raise TypeError(f"{self.__class__.__name__} must define 'name'")
        if not hasattr(self, 'description') or not self.description:
            raise TypeError(f"{self.__class__.__name__} must define 'description'")

    @abstractmethod
    def analyse(self, samples: np.ndarray, sample_rate: int) -> AnalysisResult:
        """
        Perform analysis on audio data.

        Args:
            samples: Audio samples as numpy array
            sample_rate: Sample rate in Hz

        Returns:
            AnalysisResult containing the analysis output
        """
        pass
