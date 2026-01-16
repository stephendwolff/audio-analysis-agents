"""
API Views for audio upload and track management.
"""

import uuid
from pathlib import Path

from django.conf import settings
from rest_framework import status
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from .auth import APIKeyAuthentication


# In-memory track storage (for demo - use database in production)
TRACKS: dict[str, dict] = {}


class TrackUploadView(APIView):
    """
    Upload an audio file.

    POST /api/tracks/
    Headers: X-API-Key: your-key
    Body: multipart/form-data with 'file' field

    Returns:
        {
            "track_id": "uuid",
            "filename": "original.wav",
            "path": "/media/uploads/uuid.wav"
        }
    """

    authentication_classes = [APIKeyAuthentication]
    parser_classes = [MultiPartParser]

    def post(self, request):
        if "file" not in request.FILES:
            return Response(
                {"error": "No file provided"},
                status=status.HTTP_400_BAD_REQUEST
            )

        uploaded_file = request.FILES["file"]

        # Validate file extension
        allowed_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
        ext = Path(uploaded_file.name).suffix.lower()
        if ext not in allowed_extensions:
            return Response(
                {"error": f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Generate track ID and save file
        track_id = str(uuid.uuid4())
        filename = f"{track_id}{ext}"
        filepath = settings.MEDIA_ROOT / filename

        # Ensure upload directory exists
        settings.MEDIA_ROOT.mkdir(parents=True, exist_ok=True)

        # Save file
        with open(filepath, "wb") as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)

        # Store track info
        TRACKS[track_id] = {
            "track_id": track_id,
            "original_filename": uploaded_file.name,
            "filename": filename,
            "path": str(filepath),
            "size": uploaded_file.size,
        }

        return Response({
            "track_id": track_id,
            "filename": uploaded_file.name,
            "path": f"{settings.MEDIA_URL}{filename}",
        }, status=status.HTTP_201_CREATED)


class TrackDetailView(APIView):
    """
    Get or delete a track.

    GET /api/tracks/{track_id}/
    DELETE /api/tracks/{track_id}/
    """

    authentication_classes = [APIKeyAuthentication]

    def get(self, request, track_id):
        if track_id not in TRACKS:
            return Response(
                {"error": "Track not found"},
                status=status.HTTP_404_NOT_FOUND
            )

        return Response(TRACKS[track_id])

    def delete(self, request, track_id):
        if track_id not in TRACKS:
            return Response(
                {"error": "Track not found"},
                status=status.HTTP_404_NOT_FOUND
            )

        # Delete file
        track = TRACKS[track_id]
        filepath = Path(track["path"])
        if filepath.exists():
            filepath.unlink()

        # Remove from storage
        del TRACKS[track_id]

        return Response(status=status.HTTP_204_NO_CONTENT)


class TrackListView(APIView):
    """
    List all tracks.

    GET /api/tracks/
    """

    authentication_classes = [APIKeyAuthentication]

    def get(self, request):
        return Response(list(TRACKS.values()))


def get_track_path(track_id: str) -> str | None:
    """Get the file path for a track ID. Used by WebSocket consumer."""
    if track_id in TRACKS:
        return TRACKS[track_id]["path"]
    return None
