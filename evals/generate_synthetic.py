"""
Synthetic Audio Generator for Eval Suite

Generates deterministic test audio files with known properties.
Each file is designed to test specific analysis agents.

Run with: python evals/generate_synthetic.py
"""

import numpy as np
import soundfile as sf
from pathlib import Path

# Output directory for generated files
FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_RATE = 22050  # Match the default analysis sample rate


def generate_sine_wave(frequency: float, duration: float, amplitude: float = 0.8) -> np.ndarray:
    """Generate a pure sine wave."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def generate_click_track(bpm: float, duration: float, click_duration: float = 0.02) -> np.ndarray:
    """Generate a click track at specified BPM."""
    samples = np.zeros(int(SAMPLE_RATE * duration), dtype=np.float32)
    beat_interval = 60.0 / bpm  # seconds between beats
    click_samples = int(SAMPLE_RATE * click_duration)

    # Create a click sound (short burst)
    click = np.sin(2 * np.pi * 1000 * np.linspace(0, click_duration, click_samples))
    click *= np.exp(-np.linspace(0, 5, click_samples))  # Decay envelope
    click = click.astype(np.float32)

    # Place clicks at beat positions
    current_time = 0.0
    while current_time < duration:
        start_idx = int(current_time * SAMPLE_RATE)
        end_idx = min(start_idx + len(click), len(samples))
        samples[start_idx:end_idx] += click[:end_idx - start_idx]
        current_time += beat_interval

    # Normalize
    if np.max(np.abs(samples)) > 0:
        samples = samples / np.max(np.abs(samples)) * 0.8

    return samples


def generate_fade_out(duration: float, start_amplitude: float = 0.9) -> np.ndarray:
    """Generate noise that fades from loud to quiet."""
    samples = np.random.randn(int(SAMPLE_RATE * duration)).astype(np.float32)

    # Apply fade envelope
    envelope = np.linspace(start_amplitude, 0.05, len(samples))
    samples *= envelope

    return samples


def generate_silence(duration: float) -> np.ndarray:
    """Generate silence (for edge case testing)."""
    return np.zeros(int(SAMPLE_RATE * duration), dtype=np.float32)


def generate_complex(duration: float) -> np.ndarray:
    """Generate audio with multiple characteristics: tone + rhythm + dynamics."""
    # Base tone at 440Hz
    tone = generate_sine_wave(440, duration, amplitude=0.3)

    # Add rhythmic element at 120 BPM
    clicks = generate_click_track(120, duration) * 0.5

    # Add dynamic variation (volume envelope)
    envelope = np.concatenate([
        np.linspace(0.3, 1.0, int(len(tone) * 0.3)),  # Fade in
        np.ones(int(len(tone) * 0.4)),                  # Sustain
        np.linspace(1.0, 0.2, int(len(tone) * 0.3)),  # Fade out
    ])
    # Adjust envelope length to match
    envelope = np.interp(
        np.linspace(0, 1, len(tone)),
        np.linspace(0, 1, len(envelope)),
        envelope
    ).astype(np.float32)

    combined = (tone + clicks) * envelope

    # Normalize
    if np.max(np.abs(combined)) > 0:
        combined = combined / np.max(np.abs(combined)) * 0.8

    return combined


# Define all test fixtures to generate
FIXTURES = [
    {
        "name": "click_120bpm.wav",
        "generator": lambda: generate_click_track(bpm=120, duration=5.0),
        "properties": {"bpm": 120, "duration": 5.0},
        "description": "Click track at 120 BPM - tests rhythm detection",
    },
    {
        "name": "click_90bpm.wav",
        "generator": lambda: generate_click_track(bpm=90, duration=5.0),
        "properties": {"bpm": 90, "duration": 5.0},
        "description": "Click track at 90 BPM - tests rhythm detection at different tempo",
    },
    {
        "name": "sine_440hz.wav",
        "generator": lambda: generate_sine_wave(frequency=440, duration=3.0),
        "properties": {"frequency_hz": 440, "duration": 3.0},
        "description": "Pure 440Hz sine wave - tests spectral analysis",
    },
    {
        "name": "sine_1000hz.wav",
        "generator": lambda: generate_sine_wave(frequency=1000, duration=3.0),
        "properties": {"frequency_hz": 1000, "duration": 3.0},
        "description": "Pure 1kHz sine wave - tests spectral analysis at different frequency",
    },
    {
        "name": "fade_out.wav",
        "generator": lambda: generate_fade_out(duration=4.0),
        "properties": {"duration": 4.0, "dynamics": "fade_out"},
        "description": "Noise fading from loud to quiet - tests temporal/dynamics analysis",
    },
    {
        "name": "silence.wav",
        "generator": lambda: generate_silence(duration=2.0),
        "properties": {"duration": 2.0, "is_silent": True},
        "description": "Silence - edge case testing",
    },
    {
        "name": "complex.wav",
        "generator": lambda: generate_complex(duration=5.0),
        "properties": {"frequency_hz": 440, "bpm": 120, "duration": 5.0, "dynamics": "envelope"},
        "description": "Complex audio with tone, rhythm, and dynamics - tests multi-agent scenarios",
    },
]


def generate_all():
    """Generate all synthetic test fixtures."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic audio fixtures in {FIXTURES_DIR}")
    print(f"Sample rate: {SAMPLE_RATE} Hz\n")

    for fixture in FIXTURES:
        filepath = FIXTURES_DIR / fixture["name"]
        samples = fixture["generator"]()

        sf.write(filepath, samples, SAMPLE_RATE)

        duration = len(samples) / SAMPLE_RATE
        print(f"  {fixture['name']}")
        print(f"    Duration: {duration:.2f}s")
        print(f"    Properties: {fixture['properties']}")
        print(f"    Description: {fixture['description']}\n")

    print(f"Generated {len(FIXTURES)} fixtures")


if __name__ == "__main__":
    generate_all()
