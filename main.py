import argparse
import pickle

from src.data_loader import DataLoader
from src.eval_precision import evalPrecision
from src.linear_regression import LinearRegression as myLinearRegression

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from src.plot_visualization import plotVisualization


def main(args):
    # 데이터 로드
    data_loader = DataLoader(data_path_prefix="data", model_path_prefix="model")
    data = data_loader.loadData(args.data_path)
    x = data.iloc[:, :-1].values  # 마지막 열 제외한 모든 열 선택
    y = data.iloc[:, -1].values  # 마지막 열 선택

    # sklearn 모델 테스트
    sklearn_model = LinearRegression()
    sklearn_model.fit(x, y)
    sklearn_w = sklearn_model.coef_
    sklearn_b = sklearn_model.intercept_
    sklearn_y_pred = np.dot(x, sklearn_w) + sklearn_b
    evalPrecision(y, sklearn_y_pred, 'sklearn', verbose=True)

    # 내 모델 테스트
    scaler = StandardScaler()
    scaler.fit(x)
    reg_x = scaler.transform(x)
    my_model = myLinearRegression("myModel", verbose=True)
    my_model.fit(reg_x, y, args.learning_rate, args.epochs)

    # 모델과 스케일러 함께 저장
    with open('model/linear_regression_with_scaler.pkl', 'wb') as f:
        pickle.dump((scaler, my_model), f)

    # 불러오기
    loaded_scaler, loaded_model = data_loader.loadData('linear_regression_with_scaler.pkl')
    my_w, my_b = loaded_model.getParams()
    reg_x = loaded_scaler.transform(x)
    my_y_pred = np.dot(reg_x, my_w) + my_b
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
