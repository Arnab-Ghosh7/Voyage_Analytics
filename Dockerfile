# syntax=docker/dockerfile:1
#
# Voyage Analytics — prediction API image (project objective #3)
#
# Multi-stage: dependencies are compiled in a builder stage and only the
# resulting site-packages are copied into the runtime image, so build tools
# never ship to production.
#
#   docker build -t voyage-analytics-api .
#   docker run -p 5000:5000 voyage-analytics-api

# ----------------------------------------------------------------- builder ---
FROM python:3.12-slim AS builder

WORKDIR /build

# Build-only toolchain; discarded with this stage.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements-serve.txt .

# --prefix keeps everything in one tree that the runtime stage can copy wholesale.
RUN pip install --no-cache-dir --prefix=/install -r requirements-serve.txt


# ----------------------------------------------------------------- runtime ---
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="Voyage Analytics Prediction API" \
      org.opencontainers.image.description="Flight price, gender and hotel recommendation models" \
      org.opencontainers.image.source="https://github.com/your-org/voyage-analytics"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=5000

COPY --from=builder /install /usr/local

WORKDIR /app

# Only what the API actually needs. Notebooks, raw data and the training stack
# are excluded via .dockerignore.
COPY src/ ./src/
COPY models/ ./models/
COPY requirements.txt ./

# Run as a non-root user: a container that only serves predictions has no reason
# to be able to write to its own filesystem as root.
RUN useradd --create-home --shell /bin/bash --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 5000

# The API loads and warms every model at import time (~8.5 s cold), so the probe
# must allow for that before reporting healthy — otherwise an orchestrator will
# route traffic to a pod that is alive but slow.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:5000/health', timeout=4).status==200 else 1)"

# waitress, not Flask's dev server: the built-in server is single-threaded and
# explicitly not intended for production traffic.
CMD ["waitress-serve", "--host=0.0.0.0", "--port=5000", "--threads=4", \
     "src.serving.app:app"]
