FROM python:3.14-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -u 1001 appuser && mkdir -p outputs && chown appuser outputs
USER appuser

CMD ["uvicorn", "api:app", "--host", "127.0.0.1", "--port", "8000"]
