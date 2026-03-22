FROM python:3.11-slim

# System deps for common wheels (e.g., OpenCV)
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python deps first (better layer caching)
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy app code
COPY backend /app/

EXPOSE 8000

# Render sets $PORT automatically; default to 8000 for local docker runs.
CMD ["sh", "-c", "python manage.py runserver 0.0.0.0:${PORT:-8000}"]

