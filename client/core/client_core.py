import paho.mqtt.client as mqtt
import time
import uuid
import os
import pickle
import tensorflow as tf
from client.utils.callbacks import *
from client.utils.model_utils import create_model_MLP
from client.config.config import MQTT, CLIENT_ID, SUBSCRIBE_TOPICS, PUBLISH_TOPICS
from client.utils.model_utils import create_model_MLP
from client.service.create_dataset import create_dataset

# Carregar Dataset
ds = create_dataset()
ds_train, ds_test = ds

_model = create_model_MLP()

# Variaveis Globais
head_model = None
start_time = None
end_time = None

device  = mqtt.Client(client_id=CLIENT_ID, callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
device.on_publish = on_publish
device.on_subscribe = on_subscribe
device.on_unsubscribe = on_unsubscribe

def train(weights_raw):
    print(f"\n[{CLIENT_ID}] Evento: Novos pesos recebidos. Iniciando treino...")
    
    global_weights = pickle.loads(weights_raw)
    _model.set_weights(global_weights)
    
    # Treino
    _model.fit(ds_train, validation_data=ds_test, epochs=1, verbose=1)
    
    # Recupera os pesos do modelo local
    local_weights = _model.get_weights()
    
    # Publicação
    payload = pickle.dumps({
        "weights": local_weights, 
        "n": len(ds_train)
    })

    return payload

# Callback de Mensagem: Aqui nasce a orientação a eventos
def on_message(client, userdata, message):
    global head_model, start_time, end_time

    if mqtt.topic_matches_sub(sub=SUBSCRIBE_TOPICS["global_weights_topic"], topic=message.topic):
        # Chamamos a função de treino sempre que o tópico global publicar algo
        # userdata.put(message.payload)
        
        weights = message.payload
        print("mensagem recebida, colocando na fila para o treinamento")
        
        payload = train(weights)

        weights_sent = device.publish(PUBLISH_TOPICS["local_weights_topic"], payload)
        weights_sent.wait_for_publish()

        print(f"[{CLIENT_ID}] Deltas enviados ao servidor.")

    elif mqtt.topic_matches_sub("command/status", message.topic):
        if message.payload == b"Fim do treinamento":
            client.unsubcribe(SUBSCRIBE_TOPICS["global_weights_topic"])
            head_model = _model.layers[0]

# Configuração do Cliente
device.on_message = on_message
device.on_connect = on_connect # Certifique-se que o on_connect faz o subscribe no tópico global

device.connect(MQTT)

print(f"[{CLIENT_ID}] Aguardando comando do servidor...")
try:
    # O loop_forever mantém o script rodando e reagindo aos on_message
    device.loop_forever()
except KeyboardInterrupt:
    print("Encerrando...")
finally:
    device.disconnect()