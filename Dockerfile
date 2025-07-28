# ---- Stage 1: Build ----
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build dependencies only
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Copy only the requirements file to leverage Docker cache
COPY requirements.txt .

RUN pip install --upgrade pip \
 && pip install --no-cache-dir --target=/app/deps -r requirements.txt

# ---- Stage 2: Final Image ----
FROM python:3.10-slim

WORKDIR /app

# Env setup
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/deps/bin:$PATH" \
    PYTHONPATH="/app/deps"

# Install only runtime system packages
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy only the installed Python packages and app source
COPY --from=builder /app/deps /app/deps
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
