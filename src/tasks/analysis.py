"""Celery tasks for audio analysis."""

import logging

from celery import shared_task
from django.utils import timezone

from src.api.models import Track
from src.agents import get_all_agents, get_agent
from src.tools.loader import load_audio
from .notifications import notify_track_progress, notify_track_status

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=3)
def analyze_track(self, track_id: str):
    """Run all registered agents on a track."""
    try:
        track = Track.objects.get(id=track_id)
    except Track.DoesNotExist:
        logger.error(f"Track {track_id} not found")
        return {"error": f"Track {track_id} not found"}

    track.status = Track.Status.ANALYZING
    track.save(update_fields=["status"])
    notify_track_status(track_id, "analyzing", "Starting analysis...")

    try:
        # Determine the audio path - use file_url if available, otherwise storage_path
        audio_path = track.file_url if track.file_url else track.storage_path
        logger.info(f"Loading audio from: {audio_path}")

        # Load audio once
        audio = load_audio(audio_path, target_sr=22050, mono=True)
        track.duration = audio.duration

        # Run each registered agent
        results = {}
        agents = get_all_agents()
        logger.info(f"Running {len(agents)} agents on track {track_id}")

        for agent in agents:
            notify_track_progress(track_id, agent.name, "running")
            logger.info(f"Running {agent.name} agent...")

            result = agent.analyse(audio.samples, audio.sample_rate)

            if result.success:
                results[agent.name] = result.data
                notify_track_progress(track_id, agent.name, "complete")
                logger.info(f"{agent.name} analysis complete")
            else:
                results[agent.name] = {"error": result.error}
                notify_track_progress(track_id, agent.name, "failed")
                logger.warning(f"{agent.name} analysis failed: {result.error}")

        track.analysis = results
        track.status = Track.Status.READY
        track.analyzed_at = timezone.now()
        track.save()
        notify_track_status(track_id, "ready", "Analysis complete")
        logger.info(f"Track {track_id} analysis complete")

        return {"success": True, "track_id": track_id}

    except Exception as e:
        logger.exception(f"Error analyzing track {track_id}: {e}")
        track.status = Track.Status.FAILED
        track.status_message = str(e)
        track.save(update_fields=["status", "status_message"])
        notify_track_status(track_id, "failed", str(e))

        # Retry with exponential backoff
        raise self.retry(exc=e, countdown=60 * (2 ** self.request.retries))


@shared_task
def analyze_track_with_agent(track_id: str, agent_name: str):
    """Run a single agent on a track (for on-demand analysis)."""
    try:
        track = Track.objects.get(id=track_id)
    except Track.DoesNotExist:
        return {"error": f"Track {track_id} not found"}

    agent = get_agent(agent_name)
    if not agent:
        return {"error": f"Unknown agent: {agent_name}"}

    try:
        audio_path = track.file_url if track.file_url else track.storage_path
        audio = load_audio(audio_path, target_sr=22050, mono=True)
        result = agent.analyse(audio.samples, audio.sample_rate)

        # Update track analysis
        if result.success:
            track.analysis[agent_name] = result.data
        else:
            track.analysis[agent_name] = {"error": result.error}

        track.save(update_fields=["analysis"])

        return result.data if result.success else {"error": result.error}

    except Exception as e:
        logger.exception(f"Error running {agent_name} on track {track_id}: {e}")
        return {"error": str(e)}
