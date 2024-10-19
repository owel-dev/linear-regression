import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def read_csv(file_path):
    data = pd.read_csv(file_path)
    return data


def data_load(file_path):
    data = read_csv(file_path)

    features = data.iloc[:, :-1].values  # 마지막 열 제외한 모든 열 선택
    labels = data.iloc[:, -1].values  # 마지막 열 선택

    # 데이터 정규화
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    labels = (labels - labels.mean()) / labels.std()

    return features, labels


def sklearn_model_result(features, labels):
    model = LinearRegression()

    model.fit(features, labels)
    weights = model.coef_
    bias = model.intercept_
    print(f'sklearn - w: {weights}, b: {bias}')

    result = model.predict(features)

    return result


def my_linear_regression(features, labels, learning_rate, epochs):
    n_samples, n_features = features.shape
    w = np.ones(n_features)
    b = 1.0

    features_len = len(features)
    for epoch in range(epochs):
        y_pred = np.dot(features, w) + b
        error = y_pred - labels

        w_grad = (1 / features_len) * np.dot(features.T, error)
        b_grad = (1 / features_len) * np.sum(error)

        # 파라미터 업데이트
        w -= learning_rate * w_grad
        b -= learning_rate * b_grad

        # 학습 과정 중간 결과 출력
        if epoch % 1000 == 0:
            loss = np.mean(error ** 2)
            print(f'Epoch {epoch}, Loss: {loss}')

    print(f'My Model - w: {w}, b: {b}')

    return w, b


x, y = data_load('data.csv')

y_pred_sklearn = sklearn_model_result(x, y)

my_w, my_b = my_linear_regression(x, y, 0.001, 10000)

# 결과 시각화
plt.scatter(x, y, color='blue', label='Actual Data')  # 실제 데이터 산점도
plt.plot(x, y_pred_sklearn, color='red', linewidth=2, label='Regression Line')  # sklearn 회귀선

y_pred_my = np.dot(x, my_w) + my_b
plt.plot(x, y_pred_my, color='green', linewidth=2, label='My Regression Line', linestyle='--')  # 나의 회귀선

plt.xlabel('Mileage (km)')
plt.ylabel('Price')
plt.title('Linear Regression Fit')

plt.legend()
plt.grid(True)
plt.show()
