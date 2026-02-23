import pandas as pd
import tensorflow as tf
import numpy as np
import random

# ==========================================================
# 1. CARREGAMENTO E PRÉ-PROCESSAMENTO (Sem sklearn)
# ==========================================================
def create_dataset():

    df = pd.read_csv("Crop_recommendation.csv")

    # Separar features (X) e labels (y)
    X = df.drop('label', axis=1)
    y = df['label']

    # --- A) Substituindo o StandardScaler ---
    # Normalização manual: (valor - média) / desvio_padrao
    # Fazemos isso coluna por coluna no dataframe de features
    X_scaled = (X - X.mean()) / X.std()

    # --- B) Substituindo o LabelEncoder ---
    # O método 'factorize' do pandas mapeia cada texto único para um número de 0 a 21
    y_encoded, classes_originais = pd.factorize(y)
    y_encoded = pd.Series(y_encoded, name='label')

    print(f"Total de classes encontradas: {len(classes_originais)}")
    print(f"Exemplo de classes mapeadas: {classes_originais[:3]} viraram 0, 1, 2")

    # Juntar tudo no dataset completo e processado
    df_completo = pd.concat([X_scaled, y_encoded], axis=1)

    # ==========================================================
    # 2. LÓGICA DO CLIENTE: ESCOLHER UMA AMOSTRA ALEATÓRIA
    # ==========================================================

    def criar_dataset_local_do_cliente(df_base, min_linhas=200, max_linhas=700):
        """
        Simula o dataset local do cliente escolhendo aleatoriamente 
        entre min_linhas e max_linhas do dataset completo.
        """
        qtd_linhas = random.randint(min_linhas, max_linhas)
        
        # replace=False garante que o cliente não pegue a mesma linha repetida para ele mesmo
        df_cliente = df_base.sample(n=qtd_linhas, replace=False)
        
        return df_cliente

    # Criando clientes para simulação
    df_cliente_1 = criar_dataset_local_do_cliente(df_completo)
    # print(f"Cliente 1 selecionou: {len(df_cliente_1)} linhas.")

    df_cliente_2 = criar_dataset_local_do_cliente(df_completo)
    # print(f"Cliente 2 selecionou: {len(df_cliente_2)} linhas.")

    # ==========================================================
    # 3. CONVERTER PARA TENSORFLOW DATASET (Pronto pro TFF/Keras)
    # ==========================================================

    def df_to_tf_dataset(dataframe, batch_size=32):
        # Transformar em matrizes (arrays NumPy)
        labels = dataframe['label'].values
        features = dataframe.drop('label', axis=1).values
        
        # Criar o formato de dataset do TensorFlow
        ds = tf.data.Dataset.from_tensor_slices((features, labels))
        # Embaralhar para o treinamento ser eficiente e dividir em batches (lotes)
        ds = ds.shuffle(buffer_size=len(dataframe)).batch(batch_size)
        
        return ds

    # Datasets prontos para alimentar a rede MLP que fizemos antes!
    tf_dataset_c1 = df_to_tf_dataset(df_cliente_1)
    tf_dataset_c2 = df_to_tf_dataset(df_cliente_2)

    print("\nConcluído! Código rodando perfeitamente apenas com Pandas e TensorFlow.")
    
    return tf_dataset_c1