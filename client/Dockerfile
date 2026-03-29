FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY callbacks.py .
COPY model_utils.py .
COPY create_dataset.py .
COPY Crop_recommendation.csv .
COPY client_Crop_dataset.py .
COPY inference.py .

CMD ["python", "client_Crop_dataset.py"]