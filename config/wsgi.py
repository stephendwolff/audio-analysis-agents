"""
WSGI config for audio-analysis-agents project.

Used by traditional WSGI servers (gunicorn without uvicorn workers).
For WebSocket support, use ASGI (config.asgi) instead.
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

application = get_wsgi_application()
