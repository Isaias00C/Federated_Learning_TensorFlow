FROM python:3.9

WORKDIR /app

RUN pip install jaxlib==0.4.14 -f https://storage.googleapis.com/jax-releases/jax_releases.html
RUN pip install jax==0.4.14

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY callbacks.py .
COPY model_utils.py .
COPY create_dataset.py .
COPY Crop_recommendation.csv .
COPY client_Crop_dataset.py .

CMD ["python", "client_Crop_dataset.py"]