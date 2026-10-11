# Image for running model-written code (WEBAGENT_CODE_SANDBOX=docker).
# Only the libraries the coding agents are allowed to use; no app code, no secrets.
#   docker build -t webagent/code-sandbox:latest backend/sandbox
FROM python:3.12-slim

RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.2.2 \
    matplotlib==3.9.2 \
    seaborn==0.13.2 \
    && useradd -u 1000 -m sandbox

USER sandbox
WORKDIR /tmp
