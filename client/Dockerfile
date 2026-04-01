FROM python:3.9-slim

WORKDIR /app

COPY client/requirements.txt ./client/
RUN pip install --no-cache-dir -r ./client/requirements.txt

COPY client/ ./client/
COPY dataset/ ./dataset/

ENV PYTHONPATH=/app

CMD ["python", "client/core/client_core.py"]