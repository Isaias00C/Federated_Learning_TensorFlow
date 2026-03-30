import os
from pathlib import Path

MQTT = "mosquitto-service"
NUM_MODELS = 10
SERVER_ID = "server"

SUBSCRIBE_TOPICS = {
    "local_weights_topic": "pesos_local/#",
    "split_inference_send_topic": "head_model",
}

PUBLISH_TOPICS = {
    "global_weights_topic": "pesos_global",
    "split_inference_receive_topic": "output_tail_model",
    "status_topic": "status"
}
