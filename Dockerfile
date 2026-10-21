FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optional: PyTorch (CPU build) is only needed when serving the GRU model.
ARG INSTALL_TORCH=0
RUN if [ "$INSTALL_TORCH" = "1" ]; then \
        pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu; \
    fi

COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

WORKDIR /app/src

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]