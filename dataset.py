import os
from typing import List, Tuple
from sklearn.preprocessing import normalize

import librosa
import numpy as np


class DataSet:
    def __init__(self, file_dir='', output_shape='', sample_rate=''):
        self.root_file_dir = file_dir
        self.label_dict = {}
        self.output_shape = output_shape
        self.sample_rate = sample_rate




    def _set_label(self):
        file_list = os.listdir(self.root_file_dir)
        self.label_dict = dict(zip(file_list, range(len(file_list))))

    def _read_data(self, file_dir, file_name):
        data, sr = librosa.load(os.path.join(self.root_file_dir, file_dir, file_name), sr=self.sample_rate)
        return data, sr

    @staticmethod
    def _normalize_data(data: np.array):
        shape = data.shape
        data_flatten = data.ravel()
        data_flatten = data_flatten / np.linalg.norm(data_flatten)
        return data_flatten.reshape(shape)


    def _mfcc_process(self, wave, sr):
        mfcc = librosa.feature.mfcc(wave, sr=sr, n_mfcc=self.output_shape[0])
        pad_width = self.output_shape[1] - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        return DataSet._normalize_data(mfcc)

    def _segment_process(self, wave, sr):
        """
        segment wav file to 2D structure
        :param wave:
        :param sr:
        :return:
        """
        full_num_needed = self.output_shape[0] * self.output_shape[1]
        if full_num_needed <= len(wave):
            wave = wave[:full_num_needed]
        else:
            pad_num = full_num_needed - len(wave)
            wave = np.pad(wave, (0, pad_num), 'mean')
        wave = wave.reshape(self.output_shape)
        return wave

    def _save_to_npy(self):
        labels = os.listdir(self.root_file_dir)
        for label in labels:
            # print(label)
            # Init mfcc vectors
            mfcc_vectors = []
            # print(label)
            wavfiles = [os.path.join(self.root_file_dir, label) + "\\" + wavfile for wavfile in
                        os.listdir(self.root_file_dir + "\\" + label)]

            for wavfile in wavfiles:
                # print(wavfile)
                mfcc = self._read_data(wavfile)
                mfcc_vectors.append(mfcc)
            np.save(os.path.join(self.root_file_dir, label) + '.npy', mfcc_vectors)

    def _process_data(self, data, process_class=0):
        data, sr = data
        if process_class == 0:
            return self._mfcc_process(data, sr=sr)
        elif process_class == 1:
            return self._segment_process(data, sr=sr)

        return data

    def get_train_data(self, process_class=0) -> Tuple[List[List[float,]], List[int,]]:
        file_list = []
        label_list = []
        for file_dir in self.label_dict.keys():
            for file in os.listdir(os.path.join(self.root_file_dir, file_dir)):
                data = self._read_data(file_dir, file)
                data = self._process_data(data, process_class)
                file_list.append(data)
                label_list.append(self.label_dict[file_dir])
        return file_list, label_list

    def get_register_data(self , path) -> List:
        """
        返回注册成员的语音
        :return:
        """
        data = librosa.load(path, sr=self.sample_rate)
        data = self._process_data(data, process_class = 1)
        return data
        pass

    def get_test_data(self) -> List:
        """
        返回需要判断的几个成员的语音
        :return:
        """
        pass


if __name__ == '__main__':
    x, y = DataSet(file_dir="", output_shape=(32, 1024), sample_rate=16000).get_train_data()
