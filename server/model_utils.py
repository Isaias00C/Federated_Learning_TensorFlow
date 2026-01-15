import tensorflow as tf

def create_model():
    """
    Modelo CNN simplificado para MNIST (28x28x1).
    Ideal para Federated Learning devido ao baixo número de parâmetros.
    """
    model = tf.keras.models.Sequential([
        # 1ª Camada Convolucional: Extração de características básicas (bordas)
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # 2ª Camada Convolucional: Características mais complexas
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # 3ª Camada Convolucional: Refinamento espacial
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Transição para camadas densas
        tf.keras.layers.Flatten(),
        
        # 4ª Camada (Hidden Dense): Interpretação das características
        tf.keras.layers.Dense(64, activation='relu'),
        
        # 5ª Camada (Output): Classificação final (0-9)
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model