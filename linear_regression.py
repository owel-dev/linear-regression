import numpy as np


class LinearRegression():
    def __init__(self, name="Default", log=False):
        self._name = name
        self._b = None
        self._w = None
        # log 여부에 따라 출력 함수를 설정
        self.log_fn = print if log else lambda *args, **kwargs: None

    def getParams(self):
        return self._w, self._b

    def fit(self, x, y, learning_rate=0.001, epochs=10000):
        n_samples, n_features = x.shape
        self._w = np.ones(n_features)
        self._b = 1.0

        for epoch in range(epochs):
            y_pred = np.dot(x, self._w) + self._b
            error = y_pred - y

            w_grad = (1 / n_samples) * np.dot(x.T, error)
            b_grad = (1 / n_samples) * np.sum(error)

            self._w -= learning_rate * w_grad
            self._b -= learning_rate * b_grad

            if epoch % (epochs // 10) == 0:
                loss = np.mean(error ** 2)
                self.log_fn(f'#{self._name} Epoch {epoch}, Loss: {loss}')

        self.log_fn(f'#{self._name} w: {self._w}, b: {self._b}')

    def predict(self, x):
        return np.dot(x, self._w) + self._b
