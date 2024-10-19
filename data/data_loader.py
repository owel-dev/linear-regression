import os
import pandas as pd
import pickle


class DataLoader:
    def __init__(self, path_prefix="."):
        # 캐시를 위한 딕셔너리 초기화
        self._cache = {}
        self._path_prefix = path_prefix

    def loadData(self, file_path):
        file_path = os.path.join(self._path_prefix, file_path)

        # 파일이 이미 캐시에 있는지 확인
        if file_path in self._cache:
            return self._cache[file_path]

        # 파일이 존재하는지 확인
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # 확장자에 따라 처리
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == '.csv':
            data = pd.read_csv(file_path)
        elif file_extension == '.pkl':
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

        self._cache[file_path] = data
        return data

    def clearCache(self):
        self._cache.clear()

    def removeCache(self, file_path):
        if file_path in self._cache:
            del self._cache[file_path]
