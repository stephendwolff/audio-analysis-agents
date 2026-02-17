"""Tests for rule-based descriptor generation."""

import pytest
from src.sketchbook.descriptors import generate_descriptors


class TestGenerateDescriptors:

    def test_fast_tempo_returns_driving(self):
        dims = {"bpm": 150, "swing": 0.0, "steadiness": 0.5, "upbeat": False}
        assert "driving" in generate_descriptors(dims)

    def test_slow_tempo_returns_laid_back(self):
        dims = {"bpm": 80, "swing": 0.0, "steadiness": 0.5, "upbeat": False}
        assert "laid-back" in generate_descriptors(dims)

    def test_moderate_tempo(self):
        dims = {"bpm": 110, "swing": 0.0, "steadiness": 0.5, "upbeat": False}
        assert "moderate-tempo" in generate_descriptors(dims)

    def test_high_swing_returns_swung(self):
        dims = {"bpm": 110, "swing": 0.6, "steadiness": 0.5, "upbeat": False}
        assert "swung" in generate_descriptors(dims)

    def test_low_swing_returns_straight(self):
        dims = {"bpm": 110, "swing": 0.05, "steadiness": 0.5, "upbeat": False}
        assert "straight" in generate_descriptors(dims)

    def test_high_steadiness_returns_steady(self):
        dims = {"bpm": 110, "swing": 0.0, "steadiness": 0.9, "upbeat": False}
        assert "steady" in generate_descriptors(dims)

    def test_low_steadiness_returns_loose(self):
        dims = {"bpm": 110, "swing": 0.0, "steadiness": 0.3, "upbeat": False}
        assert "loose" in generate_descriptors(dims)

    def test_upbeat_returns_upbeat_start(self):
        dims = {"bpm": 110, "swing": 0.0, "steadiness": 0.5, "upbeat": True}
        assert "upbeat-start" in generate_descriptors(dims)

    def test_multiple_descriptors_combine(self):
        dims = {"bpm": 80, "swing": 0.6, "steadiness": 0.3, "upbeat": False}
        result = generate_descriptors(dims)
        assert "laid-back" in result
        assert "swung" in result
        assert "loose" in result

    def test_returns_list_of_strings(self):
        dims = {"bpm": 110, "swing": 0.2, "steadiness": 0.5, "upbeat": False}
        result = generate_descriptors(dims)
        assert isinstance(result, list)
        assert all(isinstance(d, str) for d in result)
