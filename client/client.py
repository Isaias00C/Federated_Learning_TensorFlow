import paho.mqtt.client as mqtt
import time
import uuid
import tensorflow as tf
import pickle
import queue
from callbacks import *
from model_utils import create_model

# load dataset
dataset = tf.keras.datasets.mnist.load_data()
(x_train, y_train), (x_test, y_test) = dataset
x_train, x_test = x_train / 255.0, x_test / 255.0

#client setup
client_id = f"client_f{uuid.uuid4()}"
mqttBroker = "localhost"
subscribe_topic = "federated_learning/global_weights"
publish_topic = "federated_learning/local_weights"
q = queue.Queue()

device  = mqtt.Client(client_id=client_id, userdata=q, callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
device.on_publish = on_publish
device.on_subscribe = on_subscribe
device.on_unsubscribe = on_unsubscribe
device.on_message = on_message
device.on_connect = on_connect

device.connect(mqttBroker)
device.loop_start()

_model = create_model()

try:
    while True:

        # TODO: receive weights from server (global model)
        try:
            global_model_weights = q.get(timeout=60)
        except q.empty:
            continue        

        global_model_weights = pickle.loads(global_model_weights)
        _model.set_weights(global_model_weights)
        _model.compile(optimizer='adam', 
                        loss='sparse_categorical_crossentropy', 
                        metrics=['accuracy'])

        # train and fit model
        _model.fit(x_train, y_train, epochs=5)
        _, accuracy = _model.evaluate(x_test,  y_test, verbose=2)
        print(f"accuracy: {accuracy}")

        # calculating deltas
        local_model_weights = _model.get_weights()
        updated_weights = [u - i for u, i in zip(local_model_weights, global_model_weights)]

        # send weights (local model) to server 
        updated_weights = pickle.dumps(updated_weights)
        weights_to_send = device.publish(publish_topic, updated_weights)
        print(f"client published to topic {publish_topic}")
        weights_to_send.wait_for_publish()
        time.sleep(1)

        if accuracy > 0.9:
            break
except KeyboardInterrupt:
    print("disconecting...")

finally:
    device.loop_stop()
    device.disconnect()

