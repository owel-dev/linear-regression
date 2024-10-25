# 정밀도 평가 (MSE, MAE, R²)
import numpy as np


def evalPrecision(y, y_pred, name=None, verbose=False):
    mse = np.mean((y - y_pred) ** 2)

    mae = np.mean(np.abs(y - y_pred))

    y_mean = np.mean(y)
    ss_total = np.sum((y - y_mean) ** 2)  # 총 변동 (total sum of squares)
    ss_residual = np.sum((y - y_pred) ** 2)  # 잔차 제곱합 (residual sum of squares)
    r2 = 1 - (ss_residual / ss_total)

    if verbose:
        print(
            f"#{name}"
            f", Mean Squared Error: {mse:.4f}"
            f", Mean Absolute Error: {mae:.4f}"
            f", R^2 Score: {r2:.4f}\n"
            "----------------------------"
        )

    return {'mse': mse, 'mae': mae, 'r2': r2}
