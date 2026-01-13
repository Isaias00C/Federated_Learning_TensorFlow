import tensorflow as tf
import paho.mqtt.client as mqtt
import time
import uuid
import pickle

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
        client.unsubscribe("federated_learning/weights")

def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code.is_failure:
        print(f"Failed to connect: {reason_code}. loop_forever() will retry connection")
    else:
        # we should always subscribe from on_connect callback to be sure
        # our subscribed is persisted across reconnections.
        client.subscribe("federated_learning/weights")


client_id = f"client_f{uuid.uuid4()}"
mqttBroker = "localhost"
topic = "federated_learning/weights"

server = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2, client_id="client_1")
server.on_connect = on_connect
server.on_message = on_message
server.on_subscribe = on_subscribe
server.on_unsubscribe = on_unsubscribe

server.user_data_set([])
server.connect(mqttBroker)

'''
    model functions
'''
# function to create a model
def create_model():
    """
    Versão adaptada da AlexNet para MNIST (28x28x1).
    """
    model = tf.keras.models.Sequential([
        # 1ª Camada Convolucional
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # 2ª Camada Convolucional
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # 3ª Camada Convolucional (empilhada sem pool, estilo AlexNet)
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        
        # Flatten para entrar nas Fully Connected
        tf.keras.layers.Flatten(),
        
        # Camadas Escondidas (Dense)
        tf.keras.layers.Dense(256, activation='relu'), # Reduzido de 4096 para economizar memória no teste
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        
        # Saída (10 dígitos)
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

global_model = create_model()


server.loop_start()
try:

    # TODO: receive weights from devices 
    while True:
        local_model_weights = server.user_data_get()

        if local_model_weights and len(local_model_weights) > 0:
            local_model_weights = local_model_weights[0]

            if local_model_weights:
                local_model_weights = pickle.loads(local_model_weights)

                global_model.set_weights(local_model_weights)
                print(type(global_model), global_model.summary())
                break
    
    # TODO: aggregate weights on global model
    
    # TODO: send global model to devices

except KeyboardInterrupt:
    server.loop_stop()
    server.disconnect()
