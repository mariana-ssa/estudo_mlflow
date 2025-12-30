from sklearn.metrics import mean_squared_error, r2_score


def model_metrics(y_true, y_pred):

    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_true, y_pred)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }
