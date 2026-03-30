import paho.mqtt.client as mqtt
import time
import pickle
import numpy as np
from server.utils.model_utils import create_model_MLP
from server.utils.callbacks import on_subscribe, on_unsubscribe, on_connect, on_publish
from server.config.config import MQTT, NUM_MODELS, SUBSCRIBE_TOPICS, PUBLISH_TOPICS, SERVER_ID

epochs = 0
tail_model = None

def on_message(client, userdata, message):
    global epochs, tail_model

    if mqtt.topic_matches_sub(SUBSCRIBE_TOPICS["local_weights_topic"], message.topic):

        # userdata is the structure we choose to provide, here it's a list()
        msg = pickle.loads(message.payload)

        userdata.append(msg)
        print(f"quantidade de modelos locais recebidos: {len(userdata)} de {NUM_MODELS}")

        if len(userdata) == NUM_MODELS:
            epochs += 1
            total_samples = sum(item["n"] for item in userdata)
            fed_avg_weights = [np.zeros_like(w) for w in userdata[0]["weights"]]
            
            # 2. FedAvg
            for client_data in userdata:
                client_weights = client_data["weights"]
                n_k = client_data["n"]

                weight_factor = n_k / total_samples

                for layer_idx, layer_weights in enumerate(client_weights):
                    fed_avg_weights[layer_idx] += layer_weights * weight_factor

            
            global_model.set_weights(fed_avg_weights)
            print(global_model.summary())
            userdata.clear()
            
            # send global model to devices
            global_weights_sent = server.publish(PUBLISH_TOPICS["global_weights_topic"], pickle.dumps(global_model.get_weights()), retain=True)
            global_weights_sent.wait_for_publish() 

            if epochs == 100:
                server.unsubscribe(SUBSCRIBE_TOPICS["local_weights_topic"])

                tail_model = global_model.layers[1:]

                server.publish(PUBLISH_TOPICS["status_topic"], b"Fim do treinamento")

    elif message.topic == SUBSCRIBE_TOPICS["split_inference_send_topic"]:
        activations = pickle.loads(message.payload)

        if tail_model:
            predict = tail_model.predict(activations)
            print(predict)
            predict = pickle.dumps(predict)
            predict_published = server.publish(PUBLISH_TOPICS["split_inference_receive_topic"], predict)
            predict_published.wait_for_publish()


server = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2, client_id=SERVER_ID)
server.on_connect = on_connect
server.on_message = on_message
server.on_subscribe = on_subscribe
server.on_unsubscribe = on_unsubscribe
server.on_publish = on_publish

server.user_data_set([])
server.connect(MQTT)


global_model = create_model_MLP()


try:
    # send global model to devices
    global_model_weights = global_model.get_weights()
    global_model_weights = pickle.dumps(global_model_weights)

    global_weights_pushish = server.publish(PUBLISH_TOPICS["global_weights_topic"], global_model_weights, retain=True)
    global_weights_pushish.wait_for_publish()
    time.sleep(1)

    print("aguardando mensagens dos usuarios agora")
    server.loop_forever()
    
except KeyboardInterrupt:
    server.loop_stop()
    server.disconnect()

