import math

class KNN:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.min_vals = None
        self.max_vals = None

    def _normalize(self, X):
        X_norm = []
        for row in X:
            norm_row = []
            for i in range(len(row)):
                denom = (self.max_vals[i] - self.min_vals[i])
                if denom == 0:
                    norm_row.append(0)
                else:
                    norm_row.append((row[i] - self.min_vals[i]) / denom)
            X_norm.append(norm_row)
        return X_norm

    def fit(self, X, y):
        self.min_vals = [min(row[i] for row in X) for i in range(len(X[0]))]
        self.max_vals = [max(row[i] for row in X) for i in range(len(X[0]))]

        self.X_train = self._normalize(X)
        self.y_train = y

    def _euclidean_distance(self, row1, row2):
        return math.sqrt(sum((row1[i] - row2[i]) ** 2 for i in range(len(row1))))

    def predict(self, X_test):
        X_test_norm = self._normalize(X_test)
        predictions = []

        for test_row in X_test_norm:
            distances = []
            for i in range(len(self.X_train)):
                dist = self._euclidean_distance(test_row, self.X_train[i])
                distances.append((self.y_train[i], dist))

            distances.sort(key=lambda x: x[1])
            neighbors = distances[:self.k]

            total_weight = 0
            weighted_sum = 0

            for price, dist in neighbors:
                weight = 1 / (dist + 1e-5)
                weighted_sum += price * weight
                total_weight += weight

            avg_price = weighted_sum / total_weight
            predictions.append(avg_price)

        return predictions