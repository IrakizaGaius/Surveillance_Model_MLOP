# ---- Stage 1: Build wheels ----
    FROM python:3.10-slim-bookworm AS builder

    WORKDIR /app
    
    # Install required build tools
    RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg build-essential \
        && apt-get clean && rm -rf /var/lib/apt/lists/*
    
    # Upgrade pip tools
    RUN pip install --upgrade pip setuptools wheel
    
    # Copy requirements
    COPY requirements.txt .
    
    # Build wheels without using BuildKit syntax (compatible with Railway)
    RUN pip wheel --wheel-dir=/app/wheels -r requirements.txt
    
    # ---- Stage 2: Minimal runtime ----
    FROM python:3.10-slim-bookworm
    
    WORKDIR /app
    
    ENV PYTHONDONTWRITEBYTECODE=1 \
        PYTHONUNBUFFERED=1 \
        PATH="/app/deps/bin:$PATH" \
        PYTHONPATH="/app/deps"
    
    # Install only necessary runtime packages
    RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
        && apt-get clean && rm -rf /var/lib/apt/lists/*
    
    # Install built wheels into isolated target dir
    COPY --from=builder /app/wheels /wheels
    RUN pip install --no-cache-dir --target=/app/deps /wheels/* && rm -rf /wheels
    
    # Copy app code
    COPY app/ app/
    COPY src/ src/
    COPY models/ models/
    COPY requirements.txt .
    COPY README.md .
    
    # Railway uses port 8000 by default
    EXPOSE 8000
    
    # Run FastAPI app
    CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
    