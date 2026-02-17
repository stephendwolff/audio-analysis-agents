"""API views for the Musical Sketchbook analysis endpoints."""

import logging
import uuid

from django.conf import settings
from django.core.files.storage import default_storage
from rest_framework import status
from rest_framework.parsers import MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from src.agents.registry import get_all_agents
from src.tools.loader import SUPPORTED_MIME_TYPES
from .models import Fragment
from .tasks import analyse_fragment

logger = logging.getLogger(__name__)


class HealthView(APIView):
    """GET /api/health/ — server availability check."""

    permission_classes = []
    authentication_classes = []

    def get(self, request):
        try:
            agents = [a.name for a in get_all_agents()]
        except Exception:
            return Response(
                {"status": "error", "message": "Agents unavailable"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        if not agents:
            return Response(
                {"status": "error", "message": "No agents registered"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        return Response({
            "status": "ok",
            "version": getattr(settings, "SKETCHBOOK_MODEL_VERSION", "0.0.0"),
            "agents": agents,
        })


class AnalyseView(APIView):
    """POST /api/analyse/ — submit fragment. GET /api/analyse/{fragment_id}/ — poll."""

    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser]

    def post(self, request):
        audio = request.FILES.get("audio")
        mime_type = request.data.get("mime_type", "")
        fragment_id_str = request.data.get("fragment_id", "")
        duration_str = request.data.get("duration_seconds", "")

        if not audio:
            return Response(
                {"error": "validation_error", "message": "audio file is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if not fragment_id_str:
            return Response(
                {"error": "validation_error", "message": "fragment_id is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if not duration_str:
            return Response(
                {"error": "validation_error", "message": "duration_seconds is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            fragment_id = uuid.UUID(fragment_id_str)
        except ValueError:
            return Response(
                {"error": "validation_error", "message": "fragment_id must be a valid UUID"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            duration_seconds = float(duration_str)
        except (ValueError, TypeError):
            return Response(
                {"error": "validation_error", "message": "duration_seconds must be a number"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if mime_type not in SUPPORTED_MIME_TYPES:
            return Response(
                {"error": "unsupported_format", "message": f"Cannot decode {mime_type}"},
                status=status.HTTP_422_UNPROCESSABLE_ENTITY,
            )

        # Idempotency: check for existing fragment
        try:
            existing = Fragment.objects.get(fragment_id=fragment_id)
            if existing.status == Fragment.Status.COMPLETE:
                return Response({
                    "fragment_id": str(existing.fragment_id),
                    "status": "complete",
                    "model_version": existing.model_version,
                    **existing.analysis,
                })
            if existing.status in (Fragment.Status.PENDING, Fragment.Status.ANALYZING):
                return Response(
                    {"fragment_id": str(existing.fragment_id),
                     "status": "pending",
                     "poll_url": f"/api/analyse/{existing.fragment_id}/"},
                    status=status.HTTP_202_ACCEPTED,
                )
            if existing.status == Fragment.Status.FAILED:
                existing.status = Fragment.Status.PENDING
                existing.error_message = ""
                existing.save(update_fields=["status", "error_message"])
                try:
                    analyse_fragment.delay(str(fragment_id))
                except Exception:
                    existing.status = Fragment.Status.FAILED
                    existing.error_message = "Could not queue analysis"
                    existing.save(update_fields=["status", "error_message"])
                    return Response(
                        {"error": "service_unavailable", "message": "Analysis service unavailable"},
                        status=status.HTTP_503_SERVICE_UNAVAILABLE,
                    )
                return Response(
                    {"fragment_id": str(existing.fragment_id),
                     "status": "pending",
                     "poll_url": f"/api/analyse/{existing.fragment_id}/"},
                    status=status.HTTP_202_ACCEPTED,
                )
        except Fragment.DoesNotExist:
            pass

        # Save audio file
        filename = f"fragments/{fragment_id}{_ext_from_mime(mime_type)}"
        saved_path = default_storage.save(filename, audio)

        # Create fragment
        model_version = getattr(settings, "SKETCHBOOK_MODEL_VERSION", "0.0.0")
        fragment = Fragment.objects.create(
            fragment_id=fragment_id,
            user=request.user,
            audio_storage_path=saved_path,
            mime_type=mime_type,
            duration_seconds=duration_seconds,
            model_version=model_version,
        )

        try:
            analyse_fragment.delay(str(fragment_id))
        except Exception:
            fragment.status = Fragment.Status.FAILED
            fragment.error_message = "Could not queue analysis"
            fragment.save(update_fields=["status", "error_message"])
            return Response(
                {"error": "service_unavailable", "message": "Analysis service unavailable"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        return Response(
            {"fragment_id": str(fragment.fragment_id),
             "status": "pending",
             "poll_url": f"/api/analyse/{fragment.fragment_id}/"},
            status=status.HTTP_202_ACCEPTED,
        )

    def get(self, request, fragment_id):
        try:
            fragment_uuid = uuid.UUID(str(fragment_id))
            fragment = Fragment.objects.get(fragment_id=fragment_uuid)
        except (ValueError, Fragment.DoesNotExist):
            return Response(
                {"detail": "Not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        if fragment.status == Fragment.Status.COMPLETE:
            return Response({
                "fragment_id": str(fragment.fragment_id),
                "status": "complete",
                "model_version": fragment.model_version,
                **fragment.analysis,
            })

        if fragment.status == Fragment.Status.FAILED:
            return Response({
                "fragment_id": str(fragment.fragment_id),
                "status": "failed",
                "error": fragment.error_message,
            })

        return Response({
            "fragment_id": str(fragment.fragment_id),
            "status": "pending",
        })


def _ext_from_mime(mime_type: str) -> str:
    mapping = {
        "audio/wav": ".wav", "audio/wave": ".wav", "audio/x-wav": ".wav",
        "audio/mp3": ".mp3", "audio/mpeg": ".mp3",
        "audio/flac": ".flac", "audio/ogg": ".ogg",
        "audio/mp4": ".mp4", "audio/m4a": ".m4a", "audio/x-m4a": ".m4a",
        "audio/webm": ".webm", "audio/aac": ".aac",
    }
    return mapping.get(mime_type, ".bin")
