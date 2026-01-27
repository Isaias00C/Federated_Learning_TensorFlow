import tensorflow as tf
import tensorflow_privacy as tfp

def _optimizer() -> tf.optimizer:
    return tfp.privacy.DPKerasSGDOptimizer(
        l2_norm_clip=1.0,
        noise_multiplier=1.3,
        num_microbatches=1,
        learning_rate=0.05
    )

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
    
    model.compile(optimizer=_optimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model