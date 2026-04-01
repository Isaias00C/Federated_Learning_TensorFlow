import sys
import time
import pickle
import paho.mqtt.client as mqtt
import tensorflow as tf
from client.config.config import MQTT, SUBSCRIBE_TOPICS, PUBLISH_TOPICS, WEIGHTS_SAVE_PATH
from client.service.model_utils import create_model_MLP
from client.service.create_dataset import create_dataset
ds = create_dataset()
_, ds_test, _n = ds

# Variáveis globais para controle do tempo de ida e volta do MQTT
recebeu_resultado = False
tempo_final = 0

def on_message(client, userdata, message):
    global recebeu_resultado, tempo_final
    if message.topic == SUBSCRIBE_TOPICS["split_inference_receive_topic"]:
        tempo_final = time.time() # Marca o tempo exato que a resposta chegou do servidor
        recebeu_resultado = True

# Configurando o cliente MQTT para a inferência
mqtt_client = mqtt.Client(client_id="Client_Inference_Test", callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
mqtt_client.on_message = on_message
mqtt_client.connect(MQTT)
mqtt_client.subscribe(SUBSCRIBE_TOPICS["split_inference_receive_topic"])
mqtt_client.loop_start() # Inicia a thread do MQTT para escutar em background

def local_inference(data, model):
    print("Iniciando Inferencia Local...")
    start_time = time.time()
    _ = model.predict(data)
    end_time = time.time()

    print(f"Latência Local: {end_time - start_time:.4f} segundos")

def split_inference(data, model):
    global recebeu_resultado, tempo_final
    recebeu_resultado = False
    print("Iniciando Inferencia Split...")

    # 1. Separar a 'cabeça' do modelo (Ajuste os índices conforme sua arquitetura)
    # Supondo que a camada 0 e 1 fiquem no cliente
    head_model = tf.keras.Sequential(model.layers[:1]) 

    # Inicia o cronômetro (processamento local + rede + processamento servidor + rede)
    start_time = time.time()

    # 2. Processamento local (Extração de características)
    ativacoes_intermediarias = head_model.predict(data)

    # 3. Enviar para o servidor
    payload = pickle.dumps(ativacoes_intermediarias)
    mqtt_client.publish(PUBLISH_TOPICS["split_inference_send_topic"], payload)
    print("Dados enviados ao servidor. Aguardando resposta...")

    # 4. Aguardar a resposta do servidor para calcular a latência total
    while not recebeu_resultado:
        pass # Fica em loop até o callback on_message ser acionado

    print(f"Latência Split (Ida e Volta): {tempo_final - start_time:.4f} segundos")

if __name__ == "__main__":
    # Carregar os pesos treinados! (Descomente e ajuste se salvou no final do treino)
    # _model.load_weights("caminho/para/pesos_treinados.h5")

    # Pega apenas 1 batch de dados para o teste de latência
    batch_data = next(iter(ds_test.take(1)))[0] # Pega apenas o X (features), ignora o Y (labels)

    _model = create_model_MLP()
    _model.load_weights(WEIGHTS_SAVE_PATH)

    # Lógica para escolher no terminal qual inferência rodar
    if len(sys.argv) > 1:
        modo = sys.argv[1].lower()
        if modo == "local":
            local_inference(data=batch_data, model=_model)
        elif modo == "split":
            split_inference(data=batch_data, model=_model)
        else:
            print("Modo não reconhecido. Use 'local' ou 'split'.")
    else:
        print("Por favor, especifique o modo de inferência passando um argumento no terminal:")
        print("Exemplo: python script_inferencia.py local")
        print("Exemplo: python script_inferencia.py split")
        
    mqtt_client.loop_stop()
    mqtt_client.disconnect()