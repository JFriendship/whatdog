# Build stage: install dependencies
FROM python:3.12-slim AS builder
WORKDIR /app

COPY requirements-prod.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements-prod.txt

# Production stage: minimal runtime image
FROM python:3.12-slim

# Install only runtime essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY app .

# Create non-root user with proper permissions
RUN groupadd -r app && useradd -r -g app app && \
    chown -R app:app /app
USER app

EXPOSE 8080

# Use explicit entrypoint for better signal handling
ENTRYPOINT ["python", "-m", "uvicorn"]
CMD ["main:app", "--host", "0.0.0.0", "--port", "8080"]
