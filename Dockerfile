# ---- Builder Stage ----
FROM python:3.13-slim AS builder

WORKDIR /app

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Install dependencies
COPY requirements.txt .
RUN python - <<'PY' \
 && pip install --no-cache-dir -r /tmp/requirements.docker.txt
import sys

lines = open("requirements.txt", "r", encoding="utf-8").read().splitlines()
out = []
for line in lines:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        continue
    name = stripped.split(";", 1)[0].strip().lower()
    if name.startswith("tiktoken") and sys.version_info >= (3, 13):
        continue
    out.append(line)
open("/tmp/requirements.docker.txt", "w", encoding="utf-8").write("\n".join(out) + "\n")
PY

# ---- Runtime Stage ----
FROM python:3.13-slim

WORKDIR /app

# Volumes
VOLUME /app/src/logs

# Explicitly set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
ENV CREATE_LOG=True \
    CONFIG_PATH="/app/config.json"

RUN mkdir -p /app/src/logs

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY src/ ./src/

# Activate venv and set logs volume
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app/src"

# Expose the application port
EXPOSE 4000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD ["python", "-m", "aiproxy.healthcheck"]

# Run the script
CMD ["python", "-m", "aiproxy"]
