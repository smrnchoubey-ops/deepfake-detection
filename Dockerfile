FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create necessary directories
RUN mkdir -p static/frames static/heatmaps static/debug_crops Uploaded_Files instance

# Set environment variables
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 10000

# Run application
CMD ["gunicorn", "--workers", "2", "--worker-class", "sync", "--bind", "0.0.0.0:10000", "--timeout", "120", "server:app"]
