FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .

RUN pip install --no-cache-dir .

COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
