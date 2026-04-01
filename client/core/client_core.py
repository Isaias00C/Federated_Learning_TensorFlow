import os

import paho.mqtt.client as mqtt
import time
import pickle
from client.utils.callbacks import *
from client.config.config import MQTT, CLIENT_ID, SUBSCRIBE_TOPICS, PUBLISH_TOPICS, WEIGHTS_SAVE_PATH
from client.service.model_utils import create_model_MLP
from client.service.create_dataset import create_dataset
import threading

# Carregar Dataset
ds = create_dataset()
ds_train, ds_test, N_TRAIN_SAMPLES = ds

_model = create_model_MLP()

device = mqtt.Client(client_id=CLIENT_ID, callback_api_version=mqtt.CallbackAPIVersion.VERSION2)


def train_and_publish(weights_raw):
    try:
        print(f"\n[{CLIENT_ID}] Evento: Novos pesos recebidos. Iniciando treino...")
        
        global_weights = pickle.loads(weights_raw)
        _model.set_weights(global_weights)
        
        # Treino
        _model.fit(x=ds_train, validation_data=ds_test, epochs=1, verbose=2)
        
        # Recupera os pesos do modelo local
        local_weights = _model.get_weights()
        
        payload = pickle.dumps({
            "weights": local_weights, 
            "n": N_TRAIN_SAMPLES
        })

        weights_sent = device.publish(PUBLISH_TOPICS["local_weights_topic"], payload)
        weights_sent.wait_for_publish()
    except Exception as e:
        import traceback
        print(f"[{CLIENT_ID}] ❌ ERRO na thread: {type(e).__name__}: {e}")
        traceback.print_exc()

# Callback de Mensagem: Aqui nasce a orientação a eventos
def on_message(client, userdata, message):
    if mqtt.topic_matches_sub(sub=SUBSCRIBE_TOPICS["global_weights_topic"], topic=message.topic):
        weights = message.payload
        print("mensagem recebida, colocando na fila para o treinamento")

        t = threading.Thread(target=train_and_publish, args=(weights,))
        t.daemon = True
        t.start()

    elif mqtt.topic_matches_sub(SUBSCRIBE_TOPICS["status_topic"], message.topic):
        if message.payload == b"Parar Treinamento":
            print("Fim do trinamento. Pronto para inferencia")
            os.makedirs(os.path.dirname(WEIGHTS_SAVE_PATH), exist_ok=True)
            _model.save(WEIGHTS_SAVE_PATH)
            client.unsubscribe(SUBSCRIBE_TOPICS["global_weights_topic"])

# Configuração do Cliente
device.on_message = on_message
device.on_connect = on_connect
device.on_publish = on_publish
device.on_subscribe = on_subscribe
device.on_unsubscribe = on_unsubscribe

device.connect(MQTT)
device.loop_start()
print(f"[{CLIENT_ID}] Aguardando comando do servidor...")
try:
    while(1):
        time.sleep(1)
except KeyboardInterrupt:
    print("Encerrando...")
finally:
    device.loop_stop()
    device.disconnect()