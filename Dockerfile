# ---- Stage 1: Build wheels ----
    FROM python:3.10-slim-bullseye AS builder

    WORKDIR /app
    
    # Install only necessary build tools temporarily
    RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
     && apt-get clean && rm -rf /var/lib/apt/lists/*
    
    # Upgrade pip and build tools
    RUN pip install --upgrade pip setuptools wheel --no-cache-dir
    
    # Copy only requirements.txt to leverage Docker cache
    COPY requirements.txt .
    
    # Build wheels for all dependencies
    RUN pip wheel --wheel-dir=/app/wheels -r requirements.txt --no-cache-dir
    
    # ---- Stage 2: Minimal runtime image ----
    FROM python:3.10-slim-bullseye
    
    WORKDIR /app
    
    ENV PYTHONDONTWRITEBYTECODE=1 \
        PYTHONUNBUFFERED=1 \
        PATH="/app/deps/bin:$PATH" \
        PYTHONPATH="/app/deps"
    
    # Install only minimal runtime dependencies
    RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
     && apt-get clean && rm -rf /var/lib/apt/lists/*
    
    # Copy pre-built wheels from builder stage and install without dependencies
    COPY --from=builder /app/wheels /wheels
    RUN pip install --no-cache-dir --no-deps --target=/app/deps /wheels/* \
     && rm -rf /wheels
    
    # Optional: Strip unnecessary files to reduce size
    RUN find /app/deps -name '*.pyc' -delete \
     && find /app/deps -name '*.dist-info' -exec rm -rf {} + \
     && find /app/deps -type d -name 'tests' -exec rm -rf {} +
    
    # Create writable folders
    RUN mkdir -p models data
    
    # Copy application code (keep this minimal if possible)
    COPY app/ app/
    COPY src/ src/
    COPY requirements.txt .
    COPY README.md .
    COPY models/ models/
    
    # Expose FastAPI default port
    EXPOSE 8000
    
    # Run the FastAPI app
    CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
    