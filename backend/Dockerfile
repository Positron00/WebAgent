# Multi-stage build for WebAgent Backend
# Build from the backend/ folder and pass the app version from the repo-root package.json:
#   docker build --build-arg APP_VERSION=$(node -p "require('../package.json').version") -t webagent/backend .

# Stage 1: Build dependencies
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies at the known-good versions in constraints.txt
COPY requirements.txt constraints.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt -c constraints.txt

# Stage 2: Runtime
FROM python:3.12-slim

ARG APP_VERSION=0.0.0

# WEBAGENT_ENV selects config/prod.yaml (API keys required, debug off).
# NLTK data is installed at build time, so startup never downloads it.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PROMETHEUS_MULTIPROC_DIR=/tmp/webagent_prometheus \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/app:${PATH}" \
    WEBAGENT_ENV=prod \
    WEBAGENT_VERSION=${APP_VERSION} \
    WEBAGENT_NLTK_AUTO_DOWNLOAD=false \
    NLTK_DATA=/usr/local/share/nltk_data \
    HF_HOME=/app/data/hf-cache \
    HF_HUB_CACHE=/opt/hf-hub \
    SENTENCE_TRANSFORMERS_HOME=/opt/models \
    MLFLOW_TRACKING_URI=file:///app/data/mlruns \
    GIT_PYTHON_REFRESH=quiet \
    WEB_CONCURRENCY=2

# Create non-root user
RUN groupadd -g 1000 webagent && \
    useradd -u 1000 -g webagent -s /bin/bash -m webagent && \
    mkdir -p /app /app/logs /app/config /app/data && \
    chown -R webagent:webagent /app

WORKDIR /app

# Copy dependencies from builder stage
COPY --from=builder /root/.local /home/webagent/.local
ENV PATH=/home/webagent/.local/bin:$PATH

# NLTK corpora used by document extraction
RUN PYTHONUSERBASE=/home/webagent/.local python -m nltk.downloader -d /usr/local/share/nltk_data \
    punkt punkt_tab stopwords wordnet maxent_ne_chunker words

# Base MiniLM for the security guard (ContextDefender and alignment), baked in so startup never
# downloads it. Not under /app/data: that is a mounted volume. The triage encoder is the fine-tuned
# copy in models/triage/encoder/, which comes in with the code.
RUN mkdir -p /opt/models && PYTHONUSERBASE=/home/webagent/.local python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder='/opt/models')" \
    && chown -R webagent:webagent /opt/models

# Internal research's reranker, baked in for the same reason: it used to download into
# /app/data/hf-cache on the first research task, which needs Hugging Face access from the pod and
# failed when that folder wasn't writable. HF_HUB_CACHE points model lookups here.
RUN PYTHONUSERBASE=/home/webagent/.local python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L12-v2')" \
    && chown -R webagent:webagent /opt/hf-hub

# Copy application code. This includes the fine-tuned triage encoder (models/triage/encoder/),
# so startup never downloads a model.
COPY --chown=webagent:webagent . .

# Create the writable folders. COPY --chown already set the owner of everything else; a recursive
# chown/chmod here would store a second copy of /app (code and triage encoder) in a new layer.
RUN mkdir -p /app/logs /app/config /app/data && \
    chown webagent:webagent /app /app/logs /app/config && \
    chown -R webagent:webagent /app/data

# Switch to non-root user
USER webagent

# Expose port
EXPOSE 8000

# Health check (python, because the slim image has no curl)
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)" || exit 1

# Task status is shared between workers through /app/data/results, so several workers are safe.
# The startup checks run once first, so bad uat/prod settings stop the container with a clear
# message instead of uvicorn restarting failed workers forever.
# Metrics from every uvicorn worker are written to PROMETHEUS_MULTIPROC_DIR and summed by /metrics; the
# folder must exist before anything imports the app, and is emptied again before the workers start.
CMD ["sh", "-c", "mkdir -p \"$PROMETHEUS_MULTIPROC_DIR\" && python -m app.core.startup_checks && rm -rf \"$PROMETHEUS_MULTIPROC_DIR\" && mkdir -p \"$PROMETHEUS_MULTIPROC_DIR\" && exec uvicorn main:app --host 0.0.0.0 --port 8000 --workers ${WEB_CONCURRENCY}"]
