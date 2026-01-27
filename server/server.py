import tensorflow as tf
import paho.mqtt.client as mqtt
import time
import uuid
import pickle
from model_utils import create_model
from callbacks import on_subscribe, on_unsubscribe, on_message, on_connect, on_publish

NUM_MODELS = 10
server_id = f"server {uuid.uuid4()}"
MQTTBROKER = "mosquitto-service"
subscribe_topic = "federated_learning/local_weights"
publish_topic = "federated_learning/global_weights"


server = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2, client_id=server_id)
server.on_connect = on_connect
server.on_message = on_message
server.on_subscribe = on_subscribe
server.on_unsubscribe = on_unsubscribe
server.on_publish = on_publish

server.user_data_set([])
server.connect(MQTTBROKER)


global_model = create_model()

server.loop_start()

try:
    # send global model to devices
    global_model_weights = global_model.get_weights()
    global_model_weights = pickle.dumps(global_model_weights)

    global_weights_pushish = server.publish(publish_topic, global_model_weights, retain=True)
    global_weights_pushish.wait_for_publish()
    time.sleep(1)

    # receive weights from devices 
    while True:
        local_model_weights = server.user_data_get()

        if local_model_weights and len(local_model_weights) >= NUM_MODELS:
            break
    
    # aggregate weights on global model
    fedAprox_aggregation = 0
    for wights in local_model_weights:
        fedAprox_aggregation += pickle.loads(wights)

    global_model_weights += fedAprox_aggregation / NUM_MODELS
    global_model.set_weights(global_model_weights)
    print(global_model.summary())
    
except KeyboardInterrupt:
    server.loop_stop()
    server.disconnect()
