"""
API Views for audio upload and track management.
"""

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
from .models import Track


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
            "status": "pending"
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

        # Save using Django's storage backend (works with both local and S3)
        # Use a temp filename first, then rename with track ID
        import uuid
        track_id = uuid.uuid4()
        filename = f"{track_id}{ext}"
        saved_path = default_storage.save(filename, uploaded_file)
        file_url = default_storage.url(saved_path)

        # Create Track in database
        track = Track.objects.create(
            id=track_id,
            original_filename=uploaded_file.name,
            storage_path=saved_path,
            file_url=file_url,
            file_size=uploaded_file.size,
            user=request.user if request.user.is_authenticated else None,
        )

        # Queue background analysis using fat model method
        import logging
        logger = logging.getLogger(__name__)
        task_id = track.queue_analysis()
        if task_id:
            logger.info(f"Queued analysis task {task_id} for track {track.id}")
        else:
            logger.error(f"Failed to queue analysis task for track {track.id}")

        # Refresh track to get current status
        track.refresh_from_db()

        response_data = {
            "track_id": str(track.id),
            "filename": uploaded_file.name,
            "status": track.status,
        }
        if task_id:
            response_data["task_id"] = task_id

        return Response(response_data, status=status.HTTP_201_CREATED)


class TrackDetailView(APIView):
    """
    Get or delete a track.

    GET /api/tracks/{track_id}/
    DELETE /api/tracks/{track_id}/
    """

    authentication_classes = [JWTAuthentication, APIKeyAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request, track_id):
        track = Track.get_or_none(track_id)
        if track is None:
            return Response(
                {"error": "Track not found"},
                status=status.HTTP_404_NOT_FOUND
            )

        return Response({
            "track_id": str(track.id),
            "original_filename": track.original_filename,
            "storage_path": track.storage_path,
            "url": track.file_url,
            "size": track.file_size,
            "duration": track.duration,
            "status": track.status,
            "status_message": track.status_message,
            "created_at": track.created_at.isoformat(),
            "analyzed_at": track.analyzed_at.isoformat() if track.analyzed_at else None,
        })

    def delete(self, request, track_id):
        track = Track.get_or_none(track_id)
        if track is None:
            return Response(
                {"error": "Track not found"},
                status=status.HTTP_404_NOT_FOUND
            )

        # Delete file using storage backend
        if default_storage.exists(track.storage_path):
            default_storage.delete(track.storage_path)

        # Delete from database
        track.delete()

        return Response(status=status.HTTP_204_NO_CONTENT)


class TrackListView(APIView):
    """
    List all tracks.

    GET /api/tracks/list/
    """

    authentication_classes = [JWTAuthentication, APIKeyAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        # Filter by user if authenticated
        if request.user.is_authenticated:
            tracks = Track.objects.filter(user=request.user)
        else:
            tracks = Track.objects.none()

        return Response([
            {
                "track_id": str(t.id),
                "original_filename": t.original_filename,
                "status": t.status,
                "duration": t.duration,
                "created_at": t.created_at.isoformat(),
            }
            for t in tracks
        ])
