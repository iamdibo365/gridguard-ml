# ── STAGE 1: BUILDER ──────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── STAGE 2: RUNTIME ──────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS runtime

WORKDIR /app

COPY --from=builder /install /usr/local
COPY app/ ./app/

# Bake model into image — required for Kubernetes pods
COPY model/ ./model/

RUN useradd -m -u 1000 gridguard && \
    mkdir -p /app/logs && \
    chown -R gridguard:gridguard /app
USER gridguard

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

EXPOSE 8000
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2"]