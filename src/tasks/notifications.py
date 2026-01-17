"""WebSocket notification helpers for Celery tasks."""

import logging

from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer

logger = logging.getLogger(__name__)


def notify_track_status(track_id: str, status: str, message: str):
    """Push track status update to WebSocket."""
    channel_layer = get_channel_layer()
    if channel_layer is None:
        logger.warning("No channel layer available for notifications")
        return

    try:
        async_to_sync(channel_layer.group_send)(
            f"track_{track_id}",
            {
                "type": "track.status",
                "status": status,
                "message": message,
            }
        )
    except Exception as e:
        logger.warning(f"Failed to send track status notification: {e}")


def notify_track_progress(track_id: str, agent_name: str, state: str):
    """Push agent progress to WebSocket."""
    channel_layer = get_channel_layer()
    if channel_layer is None:
        logger.warning("No channel layer available for notifications")
        return

    try:
        async_to_sync(channel_layer.group_send)(
            f"track_{track_id}",
            {
                "type": "track.progress",
                "agent": agent_name,
                "state": state,
            }
        )
    except Exception as e:
        logger.warning(f"Failed to send track progress notification: {e}")
