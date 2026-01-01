FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml ./
COPY modelwatch ./modelwatch
RUN pip install --no-cache-dir . && useradd -r -u 10001 modelwatch
COPY config.json ./config.json
RUN mkdir -p /app/data /app/reports /app/state && chown -R modelwatch:modelwatch /app
USER modelwatch
EXPOSE 8000 8501
CMD ["uvicorn", "modelwatch.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
