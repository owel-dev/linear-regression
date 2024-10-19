import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def read_csv(file_path):
    data = pd.read_csv(file_path)
    return data


def data_load(data):
    data = read_csv('data.csv')

    features = data.iloc[:, :-1].values  # 마지막 열 제외한 모든 열 선택
    labels = data.iloc[:, -1].values  # 마지막 열 선택
    return features, labels


def sklearn_model_result(features, labels):
    model = LinearRegression()

    model.fit(features, labels)

    result = model.predict(features)
    return result


x, y = data_load('data.csv')

y_pred_sklearn = sklearn_model_result(x, y)

# 결과 시각화
plt.scatter(x, y, color='blue', label='Actual Data')  # 실제 데이터 산점도
plt.plot(x, y_pred_sklearn, color='red', linewidth=2, label='Regression Line')  # 회귀선

plt.xlabel('Mileage (km)')
plt.ylabel('Price')
plt.title('Linear Regression Fit')

plt.legend()
plt.grid(True)
plt.show()
