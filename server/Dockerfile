FROM python:3.9-slim

WORKDIR /app

COPY server/requirements.txt ./server/
RUN pip install --no-cache-dir -r ./server/requirements.txt

COPY server/ ./server/

ENV PYTHONPATH=/app

CMD ["python", "server/core/server_core.py"]