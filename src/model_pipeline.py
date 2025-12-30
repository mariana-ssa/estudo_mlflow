from split_data import split_data
from train_model import train_model
from metrics import model_metrics
import mlflow

X_train, X_test, y_train, y_test = split_data(
    path_data="./data/casas.csv",
    y_name="preco",
    size_test=0.2,
    random_state=42)

mlflow.set_experiment("projeto_casas_melhorado")
with mlflow.start_run():
    model = train_model(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = model_metrics(y_test, y_pred)

    mlflow.sklearn.log_model(model, name="linear_regression_model")
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)
