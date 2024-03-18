import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import constants


# ===============================
class Parsing:
    """Parsing and data preprocessing"""

    def __init__(self, paths_data):
        self.paths_data = paths_data
        self.data = {}
        self.load_data()
        self.expand_data()

    def load_data(self):
        for path in self.paths_data:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Файл {path} не найден!")

            file_extension = path.split('.')[-1]

            if file_extension == 'json':
                self.data[path] = pd.read_json(path)
            elif file_extension == 'csv':
                self.data[path] = pd.read_csv(path)
            elif file_extension == 'xml':
                self.data[path] = pd.read_xml(path)
            else:
                raise ValueError("Формат не поддерживается!")

    def expand_data(self):
        for _data in self.data.values():
            refs_quant = _data['refs_quant'][0]
            info_list = []
            for i in range(1, refs_quant + 1):
                info_list.append(constants.candidates_paths[i])
            _data['info'] = info_list

    def save_as_xml(self, filename):
        if not filename.endswith('.xml'):
            filename += '.xml'
        for key, value in self.data.items():
            value.to_xml(filename)

    def get_data(self):
        return self.data


parsing = Parsing(constants.data_path)
data = parsing.get_data()

print(data)
