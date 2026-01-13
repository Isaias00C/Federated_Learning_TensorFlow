import paho.mqtt.client as mqtt
import time

def on_publish(client, userdata, mid, reason_code, properties):
    pass

mqttBroker = "broker.emqx.io"
client = mqtt.Client(client_id="client_mqtt", callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
client.on_publish = on_publish
#client.user_data_set([1, 2, 3, 4, 5, 6, 7])
client.connect(mqttBroker)


while True:
    client.publish("topic_1", 100)
    print("client published to topic topic_1")
    time.sleep(1)
