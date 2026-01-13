import paho.mqtt.client as mqtt
import time
import uuid
import tensorflow as tf
import pickle

'''
    mqtt callback
'''

def on_publish(client, userdata, mid, reason_code, properties):
    print(f"client published to topic {topic} and the data is {userdata}")

client_id = f"client_f{uuid.uuid4()}"
mqttBroker = "localhost"
topic = "federated_learning/weights"

device  = mqtt.Client(client_id=client_id, callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
device.on_publish = on_publish
device.connect(mqttBroker)

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

# model created
with tf.device('/GPU:0'):
    _model = create_model()
    # TODO: receive weights from server (global model)
    
    

    _model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

    # O treinamento agora usará a GPU automaticamente
    # history = model.fit(train_dataset, epochs=10, validation_data=val_dataset)

# load dataset
dataset = tf.keras.datasets.mnist.load_data()
(x_train, y_train), (x_test, y_test) = dataset

# normalize data
x_train, x_test = x_train / 255.0, x_test / 255.0

global_model_weights = pickle.dumps(_model.get_weights())

# code for connecting with broker
device.loop_start()

# train and fit model
_model.fit(x_train, y_train, epochs=5)
results = _model.evaluate(x_test,  y_test, verbose=2)
print(results)
local_model_weights = pickle.dumps(_model.get_weights())

# send weights (local model) to server 

updated_weights = [u - i for u, i in zip(local_model_weights, global_model_weights)]

weights_to_send = device.publish(topic, updated_weights)
print(f"client published to topic {topic}")

weights_to_send.wait_for_publish()
time.sleep(1)

device.loop_stop()
device.disconnect()


