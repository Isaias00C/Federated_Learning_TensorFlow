import tensorflow as tf

def create_model_MLP(num_features=7, num_classes=22):
    model = tf.keras.Sequential([
        # Camada de entrada
        tf.keras.layers.InputLayer(input_shape=(num_features,)),
        
        # Uma camada oculta simples (para manter o modelo leve, parecido com a simplicidade do KNN)
        tf.keras.layers.Dense(64, activation='relu'),
        
        # Camada de saída com o número de culturas agrícolas (22 classes)
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
        loss='sparse_categorical_crossentropy', # Usado porque nossas labels são inteiros (0 a 21)
        metrics=['accuracy']
    )
    
    return model