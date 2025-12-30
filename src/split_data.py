import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(path_data, y_name, size_test, random_state):

    # lendo base de dados
    df = pd.read_csv(path_data)

    # Separando base em target e features
    if y_name not in df.columns:
        raise ValueError(f"Target column '{y_name}' not found in data.")

    X = df.drop(y_name, axis=1)
    y = df[y_name]

    # Separando base em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=size_test, random_state=random_state)

    return X_train, X_test, y_train, y_test
