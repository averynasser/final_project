FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /final_project

# system deps (optional, tapi aman untuk pandas/numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy app + data + api
COPY app ./app
COPY data ./data
COPY api.py ./api.py
COPY .env ./.env

# pastikan python bisa import "app"
ENV PYTHONPATH=/final_project

EXPOSE 8000

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]

