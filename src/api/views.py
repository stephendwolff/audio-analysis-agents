"""
API Views for audio upload and track management.
"""

import uuid
from pathlib import Path

from django.conf import settings
from django.core.files.storage import default_storage
from rest_framework import status
from rest_framework.parsers import MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.authentication import JWTAuthentication

from .auth import APIKeyAuthentication


# In-memory track storage (for demo - use database in production)
TRACKS: dict[str, dict] = {}


class TrackUploadView(APIView):
    """
    Upload an audio file.

    POST /api/tracks/
    Headers: Authorization: Bearer <token> OR X-API-Key: your-key
    Body: multipart/form-data with 'file' field

    Returns:
        {
            "track_id": "uuid",
            "filename": "original.wav",
            "path": "/media/uploads/uuid.wav"
        }
    """

    authentication_classes = [JWTAuthentication, APIKeyAuthentication]
    permission_classes = [IsAuthenticated]
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

        # Save using Django's storage backend (works with both local and S3)
        saved_path = default_storage.save(filename, uploaded_file)
        file_url = default_storage.url(saved_path)

        # Store track info
        TRACKS[track_id] = {
            "track_id": track_id,
            "original_filename": uploaded_file.name,
            "filename": filename,
            "storage_path": saved_path,
            "url": file_url,
            "size": uploaded_file.size,
        }

        return Response({
            "track_id": track_id,
            "filename": uploaded_file.name,
            "path": file_url,
        }, status=status.HTTP_201_CREATED)


class TrackDetailView(APIView):
    """
    Get or delete a track.

    GET /api/tracks/{track_id}/
    DELETE /api/tracks/{track_id}/
    """

    authentication_classes = [JWTAuthentication, APIKeyAuthentication]
    permission_classes = [IsAuthenticated]

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

        # Delete file using storage backend
        track = TRACKS[track_id]
        storage_path = track.get("storage_path", track.get("filename"))
        if default_storage.exists(storage_path):
            default_storage.delete(storage_path)

        # Remove from storage
        del TRACKS[track_id]

        return Response(status=status.HTTP_204_NO_CONTENT)


class TrackListView(APIView):
    """
    List all tracks.

    GET /api/tracks/list/
    """

    authentication_classes = [JWTAuthentication, APIKeyAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        return Response(list(TRACKS.values()))


def get_track_path(track_id: str) -> str | None:
    """
    Get the file path for a track ID. Used by WebSocket consumer.

    For local storage, returns the filesystem path.
    For S3 storage, returns the S3 URL (agent needs to handle this).
    """
    if track_id not in TRACKS:
        return None

    track = TRACKS[track_id]
    storage_path = track.get("storage_path", track.get("filename"))

    # For local storage, return the actual file path
    # For S3, return the URL
    if hasattr(default_storage, "path"):
        try:
            return default_storage.path(storage_path)
        except NotImplementedError:
            # S3 doesn't support path(), return URL instead
            return track.get("url")
    return track.get("url")
