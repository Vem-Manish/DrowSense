# Use a slim Python 3.10 base — mediapipe requires <=3.11
FROM python:3.10-slim

# Install system dependencies required by OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (layer cache: only re-runs pip if requirements change)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files (models, html, wav files, server)
COPY . .

# Render injects PORT at runtime; expose a default for documentation
EXPOSE 10000

# Start the FastAPI server
CMD ["python", "server.py"]
