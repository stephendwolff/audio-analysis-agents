web: python manage.py migrate --noinput && python manage.py collectstatic --noinput && gunicorn config.asgi:application -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 120
worker: celery -A src.tasks worker --loglevel=info --concurrency=2
