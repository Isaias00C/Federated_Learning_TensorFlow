import paho.mqtt.client as mqtt
import time
import uuid
import os
import pickle
import queue
import tensorflow as tf
from callbacks import *
from model_utils import create_model_MLP
from create_dataset import create_dataset

# load dataset
ds = create_dataset()
ds_train, ds_test = ds

# client setup
MQTTBROKER = "mosquitto-service"
CLIENT_ID = os.environ.get("POD_NAME", "fl-client-0")
subscribe_topic = "federated_learning/global_weights"
publish_topic = f"federated_learning/local_weights/{CLIENT_ID}"
q = queue.Queue()

_model = create_model_MLP()

device  = mqtt.Client(client_id=CLIENT_ID, userdata=q, callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
device.on_publish = on_publish
device.on_subscribe = on_subscribe
device.on_unsubscribe = on_unsubscribe

def treinar_e_enviar(weights_raw):
    print(f"\n[{CLIENT_ID}] Evento: Novos pesos recebidos. Iniciando treino...")
    
    global_weights = pickle.loads(weights_raw)
    _model.set_weights(global_weights)
    
    # Treino
    _model.fit(ds_train, validation_data=ds_test, epochs=1, verbose=1)
    
    # Cálculo de Deltas (como você já estava fazendo)
    local_weights = _model.get_weights()
    deltas = [u - i for u, i in zip(local_weights, global_weights)]
    
    # Publicação
    payload = pickle.dumps(deltas)
    device.publish(publish_topic, payload)
    print(f"[{CLIENT_ID}] Deltas enviados ao servidor.")

# Callback de Mensagem: Aqui nasce a orientação a eventos
def on_message(client, userdata, message):
    if message.topic == subscribe_topic:
        # Chamamos a função de treino sempre que o tópico global publicar algo
        treinar_e_enviar(message.payload)

# Configuração do Cliente
device = mqtt.Client(client_id=CLIENT_ID, callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
device.on_message = on_message
device.on_connect = on_connect # Certifique-se que o on_connect faz o subscribe no tópico global

device.connect(MQTTBROKER)

print(f"[{CLIENT_ID}] Aguardando comando do servidor...")
try:
    # O loop_forever mantém o script rodando e reagindo aos on_message
    device.loop_forever()
except KeyboardInterrupt:
    print("Encerrando...")
finally:
    device.disconnect()