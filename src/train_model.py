from sklearn import linear_model


def train_model(X_train, y_train):
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    return model
