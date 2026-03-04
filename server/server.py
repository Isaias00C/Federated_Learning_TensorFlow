import paho.mqtt.client as mqtt
import time
import uuid
import pickle
import numpy as np
from model_utils import create_model_MLP
from callbacks import on_subscribe, on_unsubscribe, on_connect, on_publish


NUM_MODELS = 5
MQTTBROKER = "mosquitto-service"
server_id = f"server {uuid.uuid4()}"
subscribe_topic = "federated_learning/local_weights/#"
publish_topic = "federated_learning/global_weights"


def on_message(client, userdata, message):
    # userdata is the structure we choose to provide, here it's a list()
    msg = pickle.loads(message.payload)

    userdata.append(msg)
    
    if len(userdata) == NUM_MODELS:
        fed_avg_weights = [np.copy(w) for w in userdata[0]]
        
        # 2. Soma os pesos dos modelos restantes (do segundo ao quinto)
        for client_weights in userdata[1:]:
            for layer_idx, layer_weights in enumerate(client_weights):
                fed_avg_weights[layer_idx] += layer_weights
        
        # 3. Divide pela quantidade de modelos para obter a média
        fed_avg_weights = [w / NUM_MODELS for w in fed_avg_weights]

        global_model.set_weights(fed_avg_weights)
        print(global_model.summary())
        userdata.clear()
        
        # send global model to devices
        server.publish(publish_topic, pickle.dumps(global_model.get_weights()), retain=True)



server = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2, client_id=server_id)
server.on_connect = on_connect
server.on_message = on_message
server.on_subscribe = on_subscribe
server.on_unsubscribe = on_unsubscribe
server.on_publish = on_publish

server.user_data_set([])
server.connect(MQTTBROKER)


global_model = create_model_MLP()


try:
    # send global model to devices
    global_model_weights = global_model.get_weights()
    global_model_weights = pickle.dumps(global_model_weights)

    global_weights_pushish = server.publish(publish_topic, global_model_weights, retain=True)
    global_weights_pushish.wait_for_publish()
    time.sleep(1)
    
    
except KeyboardInterrupt:
    server.loop_stop()
    server.disconnect()

server.loop_forever()