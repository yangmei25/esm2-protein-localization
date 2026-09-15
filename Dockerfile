FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_CHECKPOINT=/app/model/best_checkpoint.pt \
    MODEL_DEVICE=cpu \
    PORT=8000

WORKDIR /app

COPY requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

COPY api api
COPY src src
COPY scripts scripts
COPY web web

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/model \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["python", "scripts/start_service.py"]
