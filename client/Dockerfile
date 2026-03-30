FROM python:3.9-slim

WORKDIR /app

COPY client/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY client/config/ ./config/
COPY client/core/ ./core/
COPY client/service/ ./service/
COPY client/utils/ ./utils/

COPY dataset/dataset.csv ./dataset/

ENV PYTHONPATH=/app

CMD ["python", "core/client_core.py"]