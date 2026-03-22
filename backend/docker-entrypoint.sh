#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

export PORT="${PORT:-8000}"
export DJANGO_SETTINGS_MODULE="${DJANGO_SETTINGS_MODULE:-config.settings}"

echo "Running migrations..."
python manage.py migrate --noinput

echo "Collecting static files..."
python manage.py collectstatic --noinput || true

echo "Starting web + Celery worker + Celery beat (honcho)..."
exec honcho start -f Procfile
