import numpy as np


class Perceptron(object):
    def __init__(self, input_size, epochs=100):
        self.w = np.zeros(input_size)
        self.b = np.zeros(1)
        self.epochs = epochs

    def activation_fn(self, z):
        return 1. if z >= 0 else 0.

    def predict(self, x):
        z = self.w.T.dot(x) + self.b
        a = self.activation_fn(z)
        return a

    def fit(self, X, y):
        for _ in range(self.epochs):
            for i in range(y.shape[0]):
                a = self.predict(X[i])
                e = y[i] - a
                self.w = self.w + e * X[i]
                self.b = self.b + e


if __name__ == '__main__':
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([0, 0, 0, 1])

    perceptron = Perceptron(input_size=2)
    perceptron.fit(X, y)

    accuracy = 0.

    for i in range(y.shape[0]):
        accuracy += perceptron.predict(X[i]) == y[i]

    accuracy = accuracy / y.shape[0]
    print ("{0:4f}".format(accuracy))
    print perceptron.w
    print perceptron.b
