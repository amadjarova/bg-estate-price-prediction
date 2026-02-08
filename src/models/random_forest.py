import random

from src.models.cart import CARTRegressor

class RandomForestRegressor:
    def __init__(self, n_trees=10, max_depth=7, min_samples_split=5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def _get_bootstrap_sample(self, X, y):
        n_samples = len(X)
        indices = [random.randint(0, n_samples - 1) for _ in range(n_samples)]
        X_sample = [X[i] for i in indices]
        y_sample = [y[i] for i in indices]
        return X_sample, y_sample

    def fit(self, X, y):
        self.trees = []
        for i in range(self.n_trees):
            X_sample, y_sample = self._get_bootstrap_sample(X, y)

            tree = CARTRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)

            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_predictions = [tree.predict(X) for tree in self.trees]

        final_predictions = []
        n_samples = len(X)

        for i in range(n_samples):
            sample_preds = [tp[i] for tp in tree_predictions]
            avg_pred = sum(sample_preds) / self.n_trees
            final_predictions.append(avg_pred)

        return final_predictions