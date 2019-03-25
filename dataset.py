import os
from typing import List, Tuple

import acoustics
import librosa
import numpy as np


class DataSet:
    def __init__(self, file_dir, output_shape, sample_rate, batch_size=16, process_class=1):
        self.root_file_dir = file_dir
        self.label_dict = {}
        self.output_shape = output_shape
        self.sample_rate = sample_rate
        self.file_label = None
        self.batch_size = batch_size
        self.process_class = process_class

    def _set_label(self):
        file_list = os.listdir(self.root_file_dir)
        self.label_dict = dict(zip(file_list, range(len(file_list))))
        print(self.label_dict)

    def _read_data(self, file_path):
        data, sr = librosa.load(os.path.join(self.root_file_dir, file_path), sr=self.sample_rate)
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
        return DataSet._normalize_data(wave)

    def _add_white_noise_and_segment(self, wave, sr):
        if np.random.rand() < 0.5:
            return self._segment_process(np.add(wave, np.array(
                ((acoustics.generator.noise(sr * 90, color='white')) / 3) * 5000).astype(np.int16)[
                                                      :len(wave)]), sr)
        else:
            return self._segment_process(wave, sr)

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
        # print(1)
        if process_class == 0:
            return self._mfcc_process(data, sr=sr)
        elif process_class == 1:
            return self._segment_process(data, sr=sr)
        elif process_class == 2:
            return self._add_white_noise_and_segment(data, sr)

        return data

    def get_train_data(self, process_class=0) -> Tuple[List[List[float,]], List[int,]]:
        self._set_label()
        file_list = []
        label_list = []
        for file_dir in self.label_dict.keys():
            for file in os.listdir(os.path.join(self.root_file_dir, file_dir)):
                data = self._read_data(os.path.join(file_dir, file))
                data = self._process_data(data, process_class)
                file_list.append(data)
                label_list.append(self.label_dict[file_dir])
        return file_list, label_list

    def _set_file_label(self, process_class=1):
        file_list, label_list = self.get_train_data(process_class=process_class)
        file_label = [[] for _ in range(len(set(label_list)))]
        for file, label in zip(file_list, label_list):
            file_label[label].append(file)

        for i in range(len(file_label)):
            file_label[i] = file_label[i][:496]
        self.file_label = np.array(file_label)

    def get_triplet_batch(self, batch_size=16):
        if self.file_label is None:
            self._set_file_label(self.process_class)
        while True:
            user_input_label = np.random.randint(self.file_label.shape[0], size=batch_size)
            user_input_index = np.random.randint(self.file_label.shape[1], size=batch_size)

            user_input = self.file_label[user_input_label, user_input_index]
            positive_input = self.file_label[
                user_input_label, np.random.randint(self.file_label.shape[1], size=batch_size)]
            negative_input = self.file_label[
                np.random.randint(self.file_label.shape[0], size=batch_size), np.random.randint(
                    self.file_label.shape[1], size=batch_size)]
            yield [user_input, positive_input, negative_input], np.ones((batch_size, ))

    def get_register_data(self, path, process_class=1) -> List:
        """
        返回注册成员的语音
        :return:
        """
        data = librosa.load(path, sr=self.sample_rate)
        data = self._process_data(data, process_class=process_class)
        return data

    def get_test_data(self, path, process_class=1) -> List:
        data = librosa.load(path, sr=self.sample_rate)
        data = self._process_data(data, process_class=process_class)
        return data


if __name__ == '__main__':
    x, y = DataSet(file_dir="", output_shape=(32, 1024), sample_rate=16000).get_train_data()
