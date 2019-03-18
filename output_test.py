import argparse
import pandas as pd
import os
from dataset import DataSet
import numpy as np
from model import load_model
import keras
parser = argparse.ArgumentParser("speaker recognition", fromfile_prefix_chars='@')
parser.add_argument('--file_dir', type=str, help='Directory of test data.')
parser.add_argument('--model_path', type=str, help='Directory to load model.')
parser.add_argument('-sr', '--sample_rate', type=int, default=16000, help='sample rate of wave')

args = parser.parse_args()
root_file = args.file_dir
data_path = os.path.join(root_file , "data")
model_path = args.model_path
sample_rate = args.sample_rate


def get_group_feature():
    model = load_model(model_path)
    data = pd.read_csv(os.path.join(root_file, "enrollment.csv"))
    dataset = DataSet(sample_rate=sample_rate)
    for name , group in data.groupby('GroupID' ,as_index=False):
        #print(group)

        for person , file in group.groupby('SpeakerID'):
            for i in file['FileID'].values:
                #print(model.predict(wav2mfcc(os.path.join(data_path, i+'.wav'))))
                model.predict(dataset.get_register_data(os.path.join(data_path, i+'.wav')))



            #print(model.fit(wav2mfcc(os.path.join(data_path , str(file['FileID'])))))
            # np.append(arr , model.fit(wav2mfcc(os.path.join(data_path , file['FileID']))))

        #print( avg(arr))








def avg(arr):
    print(arr)







def save_test():
    """
    保存测试集
    :return:
    """
    return

if __name__ == '__main__':
    get_group_feature()
