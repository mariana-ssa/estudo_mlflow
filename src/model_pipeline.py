from split_data import split_data
from train_model import train_model
from metrics import model_metrics
import mlflow
import argparse


# Definição da funcao para acessar os argumentos da linha de comando
def parse_args():

    parser = argparse.ArgumentParser(description="Modelagem Projeto Alura")
    parser.add_argument("--path_data", type=str, default="./data/casas.csv",
                        help="Caminho do dataset")
    parser.add_argument("--y_name", type=str, default="preco",
                        help="Nome da coluna target")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Proporção do dataset para o conjunto de teste")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Semente para reprodutibilidade")
    parser.add_argument("--experiment_name", type=str,
                        default="projeto_casas_melhorado",
                        help="Nome do experimento MLflow")
    return parser.parse_args()


# Acessando os argumentos da linha de comando
inputs = parse_args()
path_data = inputs.path_data
y_name = inputs.y_name
test_size = inputs.test_size
random_state = inputs.random_state
experiment_name = inputs.experiment_name

# Executando a separação dos dados em treino e teste
X_train, X_test, y_train, y_test = split_data(
    path_data=path_data,
    y_name=y_name,
    size_test=test_size,
    random_state=random_state)

# Configurando o MLflow e iniciando o experimento
mlflow.set_experiment(experiment_name)
with mlflow.start_run():
    # Efetuando o treino do modelo
    model = train_model(X_train, y_train)

    # Realizando as predições no conjunto de teste
    y_pred = model.predict(X_test)

    # Calculando as métricas de avaliação do modelo
    metrics = model_metrics(y_test, y_pred)

    # Registrando o modelo e as métricas no MLflow
    mlflow.sklearn.log_model(model, name="linear_regression_model")
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)
