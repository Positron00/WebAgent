# Multi-stage build for WebAgent Backend
# Stage 1: Build dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/app:${PATH}" \
    ENVIRONMENT=production

# Create non-root user
RUN groupadd -g 1000 webagent && \
    useradd -u 1000 -g webagent -s /bin/bash -m webagent && \
    mkdir -p /app /app/logs /app/config && \
    chown -R webagent:webagent /app

WORKDIR /app

# Copy dependencies from builder stage
COPY --from=builder /root/.local /home/webagent/.local
ENV PATH=/home/webagent/.local/bin:$PATH

# Copy application code
COPY --chown=webagent:webagent . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/logs /app/config && \
    chown -R webagent:webagent /app && \
    chmod -R 755 /app

# Switch to non-root user
USER webagent

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"] 