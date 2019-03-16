import os
from typing import List, Tuple
import numpy as np
import librosa


class DataSet:
    sample_rate = 16000
    max_pad_len = 0
    def __init__(self, file_dir , max_pad_len):
        self.root_file_dir = file_dir
        self.label_dict = {}
        self.max_pad_len = max_pad_len;

    def _set_label(self):
        file_list = os.listdir(self.root_file_dir)
        self.label_dict = dict(zip(file_list, range(len(file_list))))

    def _read_data(self, file_dir):
        data, sr = librosa.load(os.path.join(self.root_file_dir, file_dir), sr=DataSet.sample_rate)
        mfcc = librosa.feature.mfcc(data, sr=DataSet.sample_rate)
        #print(mfcc.shape)
        pad_width = self.max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        return mfcc



    def _process_data(self):
        labels = os.listdir(self.root_file_dir)
        for label in labels:
            # print(label)
            # Init mfcc vectors
            mfcc_vectors = []
            #print(label)
            wavfiles = [os.path.join(self.root_file_dir , label) + "\\" + wavfile for wavfile in os.listdir(self.root_file_dir + "\\" + label)]

            for wavfile in wavfiles:
                #print(wavfile)
                mfcc = self._read_data(wavfile)
                mfcc_vectors.append(mfcc)
            np.save(os.path.join(self.root_file_dir , label) + '.npy', mfcc_vectors)
        return 1



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


test = DataSet(r"C:\Users\18140\Desktop\af2019-sr-devset-20190312\testMutDir" , 200) #输入根目录  暂定 200 最大
test._process_data() #调用 直接跑所有文件夹所有文件 在当前目录下对每个文件夹的数据集 生成对应的npy
# print(test._read_data(r"00df05c18b3ad92648119e8ad06c7fc7.wav").shape)
# print(test._read_data(r"0a26ecf9ef944bb4a82ad829c56edba2.wav").shape)