# ---- Stage 1: Build wheels ----
    FROM python:3.10-slim-bookworm AS builder

    WORKDIR /app
    
    # Install build tools required for some Python packages
    RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        build-essential \
        && apt-get clean && rm -rf /var/lib/apt/lists/*
    
    RUN pip install --upgrade pip setuptools wheel
    
    # Copy only the requirements for better caching
    COPY requirements.txt .
    
    # Build wheels (no BuildKit to avoid cache mount errors)
    RUN pip wheel --wheel-dir=/app/wheels -r requirements.txt
    
    # ---- Stage 2: Minimal runtime image ----
    FROM python:3.10-slim-bookworm
    
    WORKDIR /app
    
    ENV PYTHONDONTWRITEBYTECODE=1 \
        PYTHONUNBUFFERED=1 \
        PATH="/app/deps/bin:$PATH" \
        PYTHONPATH="/app/deps"
    
    # Install only necessary runtime packages
    RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        && apt-get clean && rm -rf /var/lib/apt/lists/*
    
    # Install the pre-built wheels in a separate target dir
    COPY --from=builder /app/wheels /wheels
    RUN pip install --no-cache-dir --target=/app/deps /wheels/* && rm -rf /wheels
    
    # Create writable folders for model and data (support retraining)
    RUN mkdir -p models data
    
    # Copy application code
    COPY app/ app/
    COPY src/ src/
    COPY requirements.txt .
    COPY README.md .
    
    # Expose default port for Uvicorn
    EXPOSE 8000
    
    # Launch the FastAPI app
    CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
    