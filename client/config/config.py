import os
from pathlib import Path

MQTT = "mosquitto-service"
CLIENT_ID = os.environ.get("POD_NAME", "fl-client-0")

SUBSCRIBE_TOPICS = {
    "global_weights_topic": "pesos_global",
    "split_inference_receive_topic": "output_tail_model",
    "status_topic": "status"
}

PUBLISH_TOPICS = {
    "local_weights_topic": f"pesos_local/{CLIENT_ID}",
    "split_inference_send_topic": "head_model"
}

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = Path("dataset") / "dataset.csv"

WEIGHTS_SAVE_PATH = BASE_DIR / "local_weights" / "weights.keras"