import tensorflow as tf
import paho.mqtt.client as mqtt
import time
import uuid
import pickle
from model_utils import create_model

'''
    mqtt callback
'''

def on_subscribe(client, userdata, mid, reason_code_list, properties):
    if reason_code_list[0].is_failure:
        print(f"Broker rejected you subscription: {reason_code_list[0]}")
    else:
        print(f"Broker granted the following QoS: {reason_code_list[0].value}")

def on_unsubscribe(client, userdata, mid, reason_code_list, properties):
    # Be careful, the reason_code_list is only present in MQTTv5.
    # In MQTTv3 it will always be empty
    if len(reason_code_list) == 0 or not reason_code_list[0].is_failure:
        print("unsubscribe succeeded (if SUBACK is received in MQTTv3 it success)")
    else:
        print(f"Broker replied with failure: {reason_code_list[0]}")
    client.disconnect()

def on_message(client, userdata, message):
    # userdata is the structure we choose to provide, here it's a list()
    userdata.append(message.payload)
    # We only want to process 10 messages
    if len(userdata) >= 10:
        client.unsubscribe("federated_learning/local_weights")

def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code.is_failure:
        print(f"Failed to connect: {reason_code}. loop_forever() will retry connection")
    else:
        # we should always subscribe from on_connect callback to be sure
        # our subscribed is persisted across reconnections.
        client.subscribe("federated_learning/local_weights")

def on_publish(client, userdata, mid, reason_code, properties):
    print(f"{client} published to topic {publish_topic} and the data is {userdata}")


server_id = f"server {uuid.uuid4()}"
mqttBroker = "localhost"
subscribe_topic = "federated_learning/local_weights"
publish_topic = "federated_learning/global_weights"


server = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2, client_id=server_id)
server.on_connect = on_connect
server.on_message = on_message
server.on_subscribe = on_subscribe
server.on_unsubscribe = on_unsubscribe
server.on_publish = on_publish

server.user_data_set([])
server.connect(mqttBroker)


global_model = create_model()


server.loop_start()
try:

    # TODO: send global model to devices
    global_model_weights = global_model.get_weights()
    global_model_weights = pickle.dumps(global_model_weights)

    global_weights_pushish = server.publish(publish_topic, global_model_weights)
    global_weights_pushish.wait_for_publish()
    time.sleep(1)

    # TODO: receive weights from devices 
    while True:
        local_model_weights = server.user_data_get()

        if local_model_weights and len(local_model_weights) >= 10:
            break
    
    # TODO: aggregate weights on global model
    
    fedAprox_aggregation = 0
    for wights in local_model_weights:
        fedAprox_aggregation += wights

    global_model_weights += fedAprox_aggregation / 10
    global_model.set_weights(global_model_weights)
    print(global_model.summary())


except KeyboardInterrupt:
    server.loop_stop()
    server.disconnect()
