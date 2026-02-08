def train_test_split(X, y, test_size=0.2):
    split_index = int(len(X) * (1 - test_size))
    return X[:split_index], X[split_index:], y[:split_index], y[split_index:]

def calculate_metrics(y_real, y_pred):
    n = len(y_real)
    mae = sum(abs(y_real[i] - y_pred[i]) for i in range(n)) / n
    mape = (sum(abs((y_real[i] - y_pred[i]) / y_real[i]) for i in range(n)) / n) * 100
    return mae, mape