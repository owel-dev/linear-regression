import os
import pandas as pd
import pickle


class DataLoader:
    def __init__(self, data_path_prefix="data", model_path_prefix="model"):
        # 캐시를 위한 딕셔너리 초기화
        self._cache = {}
        self._data_path_prefix = data_path_prefix
        self._model_path_prefix = model_path_prefix

    def loadData(self, file_path):
        data_file_path = os.path.join(self._data_path_prefix, file_path)
        model_file_path = os.path.join(self._model_path_prefix, file_path)

        # 파일이 이미 캐시에 있는지 확인
        if file_path in self._cache:
            return self._cache[file_path]

        # 파일이 존재하는지 확인
        if os.path.exists(self._data_path_prefix):
            pass
        elif os.path.exists(self._model_path_prefix):
            pass
        else:
            raise FileNotFoundError(f"File not found: {file_path}")

        # 확장자에 따라 처리
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == '.csv':
            data = pd.read_csv(data_file_path)
        elif file_extension == '.pkl':
            with open(model_file_path, 'rb') as f:
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
