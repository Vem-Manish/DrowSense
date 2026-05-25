# Use a lightweight Python 3.10 image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose Render's default internal port
EXPOSE 10000

# Start the FastAPI server using Uvicorn directly
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "10000"]