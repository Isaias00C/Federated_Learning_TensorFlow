import pandas as pd
import tensorflow as tf
import random

# ==========================================================
# 1. CARREGAMENTO E PRÉ-PROCESSAMENTO (Sem sklearn)
# ==========================================================
def create_dataset(test_split=0.2, batch_size=32):
    df = pd.read_csv("Crop_recommendation.csv")

    # 1. Pré-processamento (X e y)
    X = df.drop('label', axis=1)
    y = df['label']
    X_scaled = (X - X.mean()) / X.std()
    y_encoded, _ = pd.factorize(y)
    
    df_completo = pd.concat([X_scaled, pd.Series(y_encoded, name='label')], axis=1)

    # 2. Simulação de Cliente (Amostragem)
    qtd_linhas = random.randint(1000, 1800)
    df_cliente = df_completo.sample(n=qtd_linhas, replace=False)

    # --- NOVIDADE: Separação Treino e Teste no Pandas ---
    # Embaralhamos antes de cortar
    df_cliente = df_cliente.sample(frac=1).reset_index(drop=True)
    n_test = int(len(df_cliente) * test_split)
    
    df_test = df_cliente.iloc[:n_test]
    df_train = df_cliente.iloc[n_test:]

    # 3. Conversão para TF Dataset
    def df_to_tf_dataset(dataframe, is_train=False):
        labels = dataframe['label'].values
        features = dataframe.drop('label', axis=1).values
        # Retornamos o dataset pronto para o Keras .fit()
        ds = tf.data.Dataset.from_tensor_slices((features, labels))

        if is_train:
            ds = ds.shuffle(buffer_size=len(dataframe))
        
        return ds.batch(batch_size)

    ds_train = df_to_tf_dataset(df_train, is_train=True)
    ds_test = df_to_tf_dataset(df_test)

    print(f"Cliente com {len(df_cliente)} amostras. Treino: {len(df_train)}, Teste: {len(df_test)}")
    
    return ds_train, ds_test
