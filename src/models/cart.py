from src.utils.config  import MAX_DEPTH, MIN_SAMPLES_SPLIT

class TreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class CARTRegressor:
    def __init__(self, max_depth=MAX_DEPTH, min_samples_split=MIN_SAMPLES_SPLIT):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def predict(self, X):
        y_pred = [self._predict_sample(self.root, x) for x in X]
        return y_pred

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _mse(self, y):
        if len(y) == 0:
            return 0
        mean_y = sum(y) / len(y)
        return sum((yi - mean_y) ** 2 for yi in y) / len(y)

    def _split_dataset(self, X, y, feature_index, threshold):
        left_X, right_X, left_y, right_y = [], [], [], []
        for i in range(len(X)):
            if X[i][feature_index] <= threshold:
                left_X.append(X[i])
                left_y.append(y[i])
            else:
                right_X.append(X[i])
                right_y.append(y[i])
        return left_X, right_X, left_y, right_y

    def _possible_tresholds(self, values):
        sorted_values = sorted(set(values))
        thresholds = []
        for i in range(len(sorted_values) - 1):
            thresholds.append((sorted_values[i] + sorted_values[i + 1]) / 2)
        return thresholds

    def _best_split(self, X, y):
        best_mse = float('inf')
        best_feature_index = None
        best_threshold = None

        n_features = len(X[0])
        for feature_index in range(n_features):
            feature_values = [X[i][feature_index] for i in range(len(X))]
            thresholds = self._possible_tresholds(feature_values)

            for threshold in thresholds:
                left_X, right_X, left_y, right_y = self._split_dataset(X, y, feature_index, threshold)
                if len(left_y) == 0 or len(right_y) == 0:
                    continue

                mse = (len(left_y) * self._mse(left_y) + len(right_y) * self._mse(right_y)) / len(y)

                if mse < best_mse:
                    best_mse = mse
                    best_feature_index = feature_index
                    best_threshold = threshold
        return best_feature_index, best_threshold

    def _build_tree(self, X, y, depth):
        if len(y) < self.min_samples_split or depth >= self.max_depth:
            return TreeNode(value=sum(y) / len(y))

        feature_index, threshold = self._best_split(X, y)
        if feature_index is None:
            return TreeNode(value=sum(y) / len(y))

        left_X, right_X, left_y, right_y = self._split_dataset(X, y, feature_index, threshold)

        left_child = self._build_tree(left_X, left_y, depth + 1)
        right_child = self._build_tree(right_X, right_y, depth + 1)
        return TreeNode(feature_index=feature_index, threshold=threshold, left=left_child, right=right_child)

    def _predict_sample(self, node, sample):
        if node.value is not None:
            return node.value

        if sample[node.feature_index] <= node.threshold:
            return self._predict_sample(node.left, sample)
        else:
            return self._predict_sample(node.right, sample)