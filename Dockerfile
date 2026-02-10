FROM python:3.14.3-slim-bookworm

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    CREATE_LOG=True \
    CONFIG_PATH="/app/config.json" \
    PYTHONPATH="/app/src"

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
 && mkdir -p /app/src/logs

COPY src/ ./src/

EXPOSE 4000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD ["python", "-m", "aiproxy.healthcheck"]

CMD ["python", "-m", "aiproxy"]
