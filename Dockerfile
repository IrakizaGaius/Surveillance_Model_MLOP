# Use a lightweight Python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    && apt-get clean

# Copy dependency files
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy entire application code
COPY . .

# Expose port FastAPI will run on
EXPOSE 8000

# Default command to run the API server
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
