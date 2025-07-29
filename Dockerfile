# ---- Stage 1: Build wheels ----
    FROM python:3.10-slim-bookworm AS builder
    WORKDIR /app
    RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
        && apt-get clean && rm -rf /var/lib/apt/lists/*
    RUN pip install --upgrade pip setuptools wheel
    COPY requirements.txt .
    RUN --mount=type=cache,target=/root/.cache/pip \
        pip wheel --wheel-dir=/app/wheels -r requirements.txt
    
    # ---- Stage 2: Minimal runtime ----
    FROM python:3.10-slim-bookworm
    WORKDIR /app
    ENV PYTHONDONTWRITEBYTECODE=1 \
        PYTHONUNBUFFERED=1 \
        PATH="/app/deps/bin:$PATH" \
        PYTHONPATH="/app/deps"
    RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
        && apt-get clean && rm -rf /var/lib/apt/lists/*
    COPY --from=builder /app/wheels /wheels
    RUN pip install --no-cache-dir --target=/app/deps /wheels/* && rm -rf /wheels
    
    # Only copy what you need!
    COPY app/ app/
    COPY src/ src/
    COPY models/ models/
    COPY requirements.txt .
    COPY README.md .
    
    EXPOSE 8000
    CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]