"""Celery tasks for Musical Sketchbook fragment analysis."""

import logging

from celery import shared_task

from src.agents.rhythm import RhythmAgent
from src.agents.spectral import SpectralAgent
from src.agents.temporal import TemporalAgent
from src.tools.loader import load_audio
from .models import Fragment
from .serializers import build_analysis_result

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=3, default_retry_delay=30)
def analyse_fragment(self, fragment_id_str: str):
    """
    Run analysis agents on a fragment and store the result.

    Args:
        fragment_id_str: The fragment_id UUID as a string
    """
    try:
        fragment = Fragment.objects.get(fragment_id=fragment_id_str)
    except Fragment.DoesNotExist:
        logger.error(f"Fragment {fragment_id_str} not found")
        return

    fragment.status = Fragment.Status.ANALYZING
    fragment.save(update_fields=["status"])

    try:
        audio = load_audio(
            fragment.audio_storage_path,
            target_sr=22050,
            mono=True,
            mime_type=fragment.mime_type,
        )

        rhythm = RhythmAgent().analyse(audio.samples, audio.sample_rate)
        spectral = SpectralAgent().analyse(audio.samples, audio.sample_rate)
        temporal = TemporalAgent().analyse(audio.samples, audio.sample_rate)

        if not rhythm.success:
            raise RuntimeError(f"Rhythm analysis failed: {rhythm.error}")
        if not spectral.success:
            raise RuntimeError(f"Spectral analysis failed: {spectral.error}")
        if not temporal.success:
            raise RuntimeError(f"Temporal analysis failed: {temporal.error}")

        fragment.analysis = build_analysis_result(rhythm, spectral, temporal)
        fragment.status = Fragment.Status.COMPLETE
        fragment.save(update_fields=["analysis", "status"])

        logger.info(f"Fragment {fragment_id_str} analysis complete")

    except Exception as e:
        logger.exception(f"Error analysing fragment {fragment_id_str}: {e}")
        fragment.status = Fragment.Status.FAILED
        fragment.error_message = str(e)
        fragment.save(update_fields=["status", "error_message"])
