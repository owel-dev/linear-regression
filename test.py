import pandas as pd

# 예시 DataFrame 생성
data = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}
df = pd.DataFrame(data)

# 첫 번째 행과 두 번째 열 선택
print(df.iloc[0, 1])  # 출력: 4

# 첫 번째 행 전체 선택
print(df.iloc[0, :])
# 출력:
# A    1
# B    4
# C    7
# Name: 0, dtype: int64

# 첫 번째와 두 번째 행, A와 B열 선택
print(df.iloc[0:2, 0:2])
# 출력:
#    A  B
# 0  1  4
# 1  2  5
