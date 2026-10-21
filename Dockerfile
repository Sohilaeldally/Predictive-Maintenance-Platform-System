# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# Runtime-only dependencies (small). The cache mount keeps downloaded wheels
# between builds, so a retry after a network timeout does not start from zero.
COPY requirements-api.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=300 --retries 10 -r requirements-api.txt

# Optional: PyTorch (CPU build) is only needed when serving the GRU model.
ARG INSTALL_TORCH=0
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$INSTALL_TORCH" = "1" ]; then \
        pip install --default-timeout=300 --retries 10 torch --index-url https://download.pytorch.org/whl/cpu; \
    fi

COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

WORKDIR /app/src

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]