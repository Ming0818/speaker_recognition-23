import os
from typing import List, Tuple

import librosa


class DataSet:
    sample_rate = 16000

    def __init__(self, file_dir):
        self.root_file_dir = file_dir
        self.label_dict = {}

    def _set_label(self):
        file_list = os.listdir(self.root_file_dir)
        self.label_dict = dict(zip(file_list, range(len(file_list))))

    def _read_data(self, file_dir):
        data, sr = librosa.load(os.path.join(self.root_file_dir, file_dir), sr=DataSet.sample_rate)
        return data, sr

    def _process_data(self, data):
        return data

    def get_train_data(self) -> Tuple[List[], List[int,]]:
        file_list = []
        label_list = []
        for file_dir in self.label_dict.keys():
            data = self._read_data(file_dir)
            data = self._process_data(data)
            file_list.append(data)
            label_list.append(self.label_dict[file_dir])
        return file_list, label_list
