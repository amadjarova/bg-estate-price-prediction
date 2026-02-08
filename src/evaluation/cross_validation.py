import math

def cross_validate(model_class, X, y, folds=10, **model_params):
    fold_size = len(X) // folds
    scores = []
    epsilon = 1e-10

    for i in range(folds):
        start, end = i * fold_size, (i + 1) * fold_size

        X_test = X[start:end]
        y_test = y[start:end]

        X_train = X[:start] + X[end:]
        y_train = y[:start] + y[end:]

        model = model_class(**model_params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)


        total_percentage_error = 0
        for j in range(len(y_test)):
            actual_price = y_test[j] if y_test[j] != 0 else epsilon
            total_percentage_error += abs((actual_price - preds[j]) / actual_price)

        mape = (total_percentage_error / len(y_test)) * 100
        scores.append(100 - mape)

    return sum(scores) / len(scores)