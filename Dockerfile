# BUILD STAGE
FROM python:3.12-slim AS builder

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /build
COPY requirements-prod.txt .
RUN python -m pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.11.0 torchvision==0.26.0 \
&& python -m pip install --no-cache-dir -r requirements-prod.txt

# RUNNER STAGE
FROM python:3.12-slim AS runner

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    CHECKPOINT_PATH=/app/app/training/artifacts/checkpoints/last.ckpt \
    CLASS_NAMES_PATH=/app/app/training/artifacts/class_names.json

WORKDIR /app

RUN useradd --system --create-home --user-group appuser

COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appuser app ./app

USER appuser

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.api.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]