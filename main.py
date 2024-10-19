from eval_precision import evalPrecision
from linear_regression import LinearRegression as myLinearRegression

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from plot_visualization import plotVisualization


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

plotVisualization(x, y, sklearn_pred, my_y_pred)
