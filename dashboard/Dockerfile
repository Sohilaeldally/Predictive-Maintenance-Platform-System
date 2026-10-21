# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# The cache mount keeps downloaded wheels between builds (useful on a slow
# connection: a retry resumes instead of starting from zero).
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=300 --retries 10 -r requirements.txt

COPY app.py .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]