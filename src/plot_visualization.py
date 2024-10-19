# 결과 시각화
from matplotlib import pyplot as plt


def plotVisualization(x, y, first_y, second_y, title='Model Fit'):
    plt.scatter(x, y, color='blue', label='Actual Data')  # 실제 데이터 산점도

    plt.plot(x, first_y, color='red', linewidth=2, label='First Y')
    plt.plot(x, second_y, color='yellow', linewidth=3, label='Second Y', linestyle=':')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
