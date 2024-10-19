from eval_precision import evalPrecision
from linear_regression import LinearRegression as myLinearRegression

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


def sklearnModel(x, y):
    model = LinearRegression()

    model.fit(x, y)

    w = model.coef_
    b = model.intercept_

    print(f'#sklearn w: {w}, b: {b}')
    return w, b


x, y = data_load('data.csv')

sklearn_w, sklearn_b = sklearnModel(x, y)
sklearn_pred = np.dot(x, sklearn_w) + sklearn_b
evalPrecision(y, sklearn_pred, 'sklearn', log=True)

myModel = myLinearRegression(log=True)
myModel.fit(x, y)
my_w, my_b = myModel.getParams()
my_y_pred = np.dot(x, my_w) + my_b
evalPrecision(y, my_y_pred, 'My', log=True)

# 결과 시각화
plt.scatter(x, y, color='blue', label='Actual Data')  # 실제 데이터 산점도
plt.plot(x, sklearn_pred, color='red', linewidth=2, label='Regression Line')  # sklearn 회귀선
plt.plot(x, my_y_pred, color='green', linewidth=2, label='My Regression Line', linestyle='--')  # 나의 회귀선
plt.xlabel('Mileage (km)')
plt.ylabel('Price')
plt.title('Linear Regression Fit')
plt.legend()
plt.grid(True)
plt.show()
