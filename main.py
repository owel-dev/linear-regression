import argparse

from data.data_loader import DataLoader
from src.eval_precision import evalPrecision
from src.linear_regression import LinearRegression as myLinearRegression

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from src.plot_visualization import plotVisualization


def main(args):
    # 데이터 로드
    data_loader = DataLoader("data")
    data = data_loader.loadData(args.data_path)
    x = data.iloc[:, :-1].values  # 마지막 열 제외한 모든 열 선택
    y = data.iloc[:, -1].values  # 마지막 열 선택

    # 데이터 정규화
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    y = (y - y.mean()) / y.std()

    # sklearn 모델 테스트
    sklearn_model = LinearRegression()
    sklearn_model.fit(x, y)
    sklearn_w = sklearn_model.coef_
    sklearn_b = sklearn_model.intercept_
    sklearn_y_pred = np.dot(x, sklearn_w) + sklearn_b
    evalPrecision(y, sklearn_y_pred, 'sklearn', True)

    # 내 모델 테스트
    myModel = myLinearRegression("myModel", True)
    myModel.fit(x, y, args.learning_rate, args.epochs)
    my_w, my_b = myModel.getParams()
    my_y_pred = np.dot(x, my_w) + my_b
    evalPrecision(y, my_y_pred, 'My', True)

    # 결과 시각화
    plotVisualization(x, y, sklearn_y_pred, my_y_pred)


if __name__ == "__main__":
    # argparse를 사용해 커맨드라인 인자 처리
    parser = argparse.ArgumentParser(description="Arguments for linear regression")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data file')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs for training')
    args = parser.parse_args()
    main(args)
