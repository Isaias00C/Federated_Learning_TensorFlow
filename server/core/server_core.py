import paho.mqtt.client as mqtt
import time
import pickle
import numpy as np
import threading
from server.utils.model_utils import create_model_MLP
from server.utils.callbacks import on_subscribe, on_unsubscribe, on_connect, on_publish
from server.config.config import MQTT, NUM_MODELS, SUBSCRIBE_TOPICS, PUBLISH_TOPICS, SERVER_ID

epochs = 0
tail_model = None
global_model = create_model_MLP()


def aggregate_and_publish(userdata_snapshot):
    global epochs, tail_model

    epochs += 1
    print(f"[Servidor] Iniciando FedAvg — round {epochs}")

    total_samples = sum(item["n"] for item in userdata_snapshot)
    fed_avg_weights = [np.zeros_like(w) for w in userdata_snapshot[0]["weights"]]

    for client_data in userdata_snapshot:
        n_k = client_data["n"]
        weight_factor = n_k / total_samples
        for i, layer_weights in enumerate(client_data["weights"]):
            fed_avg_weights[i] += layer_weights * weight_factor

    global_model.set_weights(fed_avg_weights)
    print(f"[Servidor] Round {epochs} agregado.")

    payload = pickle.dumps(global_model.get_weights())
    pub = server.publish(PUBLISH_TOPICS["global_weights_topic"], payload, retain=True)
    pub.wait_for_publish()
    print(f"[Servidor] Pesos globais publicados.")

    if epochs == 100:
        server.unsubscribe(SUBSCRIBE_TOPICS["local_weights_topic"])
        server.publish(PUBLISH_TOPICS["status_topic"], b"Parar Treinamento")
        print("[Servidor] Treinamento encerrado.")

def on_message(client, userdata, message):
    if mqtt.topic_matches_sub(SUBSCRIBE_TOPICS["local_weights_topic"], message.topic):
        msg = pickle.loads(message.payload)
        userdata.append(msg)
        print(f"[Servidor] Modelos recebidos: {len(userdata)} de {NUM_MODELS}")

        if len(userdata) == NUM_MODELS:
            snapshot = list(userdata)
            userdata.clear()
            t = threading.Thread(target=aggregate_and_publish, args=(snapshot,))
            t.daemon = True
            t.start()

    elif message.topic == SUBSCRIBE_TOPICS["split_inference_send_topic"]:
        activations = pickle.loads(message.payload)
        if tail_model:
            predict = pickle.dumps(tail_model.predict(activations))
            pub = server.publish(PUBLISH_TOPICS["split_inference_receive_topic"], predict)
            pub.wait_for_publish()


server = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2, client_id=SERVER_ID)
server.on_connect = on_connect
server.on_message = on_message
server.on_subscribe = on_subscribe
server.on_unsubscribe = on_unsubscribe
server.on_publish = on_publish
server.user_data_set([])

server.connect(MQTT)
server.loop_start()  # ✅ loop antes dos publishes

global_model_weights = pickle.dumps(global_model.get_weights())

init = server.publish(topic=PUBLISH_TOPICS["status_topic"], payload=b"Iniciar Treinamento")
init.wait_for_publish()
print("[Servidor] Sinal de início enviado.")

weights_pub = server.publish(topic=PUBLISH_TOPICS["global_weights_topic"], payload=global_model_weights, retain=True)
weights_pub.wait_for_publish()
print("[Servidor] Pesos iniciais publicados. Aguardando clientes...")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Encerrando...")
finally:
    server.loop_stop()
    server.disconnect()