import numpy as np

class LinearSVC:
    def __init__(self, learning_rate=0.01, epochs=1000, regularization_param=0.01, random_seed=None):
        np.random.seed(random_seed)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.regularization_param = regularization_param
        self.weights = None
        self.bias = None

    def net_input(self, X):
        """ Compute the preactivation value"""
        return np.dot(X, self.weights) + self.bias

    def fit(self, X, y):
        """Train the model using a soft-margin SVC"""
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features)
        self.bias = 0

        # Training loop (Epochs)
        for _ in range(self.epochs):
            for i in range(n_samples):
                condition = y[i] * self.net_input(X[i]) >= 1
                if condition:
                    # If the point is correctly classified, apply regularization
                    self.weights -= self.learning_rate * (2 * self.regularization_param * self.weights)
                else:
                    # If the point is misclassified, update weights and bias
                    self.weights -= self.learning_rate * (
                                2 * self.regularization_param * self.weights - np.dot(X[i], y[i]))
                    self.bias -= self.learning_rate * y[i]


    def predict(self, X):
        """Generate predictions for input samples"""
        preactivation = self.net_input(X)
        return np.sign(preactivation)


