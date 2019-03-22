import argparse
import os

import numpy as np
import pandas as pd

from dataset import DataSet
from feature_transform import mean_vectors
from model import load_model
from feature_transform import get_mean_feature_for_device

parser = argparse.ArgumentParser("speaker recognition", fromfile_prefix_chars='@')
parser.add_argument('--file_dir', type=str, help='Directory of test data.')
parser.add_argument('--model_path', type=str, help='Directory to load model.')
parser.add_argument('-sr', '--sample_rate', type=int, default=16000, help='sample rate of wave')
parser.add_argument('-s', '--output_shape', type=int, nargs=2, default=[32, 1024], help='shape')

args = parser.parse_args()
root_file = args.file_dir
data_path = os.path.join(root_file, "data")
model_path = args.model_path
sample_rate = args.sample_rate
output_shape = args.output_shape


def get_group_feature():
    model = load_model(model_path, model_type=1)
    data = pd.read_csv(os.path.join(root_file, "enrollment.csv"))
    dataset = DataSet(file_dir='', output_shape=output_shape, sample_rate=sample_rate)

    rows_list = []
    for name, group in data.groupby('GroupID', as_index=False):

        for person, file in group.groupby('SpeakerID'):
            li = []
            for i in file['FileID'].values:
                # print(model.predict(wav2mfcc(os.path.join(data_path, i+'.wav'))))
                arr = np.array(dataset.get_register_data(os.path.join(data_path, i + '.wav')))
                li.append(model.predict(arr.reshape((1, *arr.shape))))

            personLi = [person]
            personLi.extend((mean_vectors(li)[0]))
            groupLi = [name]
            groupLi.extend(personLi)
            rows_list.append(groupLi)
    res = pd.DataFrame(rows_list)
    #print(res)
    res.to_csv(os.path.join(root_file, 'enroll.csv'))
    return rows_list


def save_test():
    """
    保存测试集
    :return:
    """
    return

def test_output(if_handle_device = False):

    model = load_model(model_path, model_type=1)
    data = pd.read_csv(os.path.join(root_file, "test.csv"))
    dataset = DataSet(file_dir='', output_shape=output_shape, sample_rate=sample_rate)
    count = 0
    rows_list = []
    for index , row in data.iterrows():
        tester = [row["GroupID"] , row["FileID"]]
        wav_data = dataset.get_test_data(os.path.join(data_path, row["FileID"] + ".wav"))
        if if_handle_device:
            if row["DeviceID"] == 1:
                device_arr = get_mean_feature_for_device(model_path=model_path, path=root_file,
                                                         output_shape=output_shape, sample_rate=sample_rate)
                arr = np.array(wav_data)
                model_predict_data = model.predict(arr.reshape((1, *arr.shape)))
                model_predict_data = model_predict_data + np.reshape(device_arr[1] - device_arr[0] , (1 , device_arr[1].shape[0]))
                #print(model_predict_data.shape)
                tester.extend(model_predict_data)
            else:
                arr = np.array(wav_data)
                model_predict_data = model.predict(arr.reshape((1, *arr.shape)))
                tester.extend(model_predict_data)

        else:
            arr = np.array(wav_data)
            model_predict_data = model.predict(arr.reshape((1, *arr.shape)))
            tester.extend(model_predict_data)


        rows_list.append(tester)
        count = count + 1
        print(count)
    pd.DataFrame(rows_list).to_csv(os.path.join(root_file , "test_output.csv"))











    #     for person, file in group.groupby('SpeakerID'):
    #         li = []
    #         for i in file['FileID'].values:
    #             # print(model.predict(wav2mfcc(os.path.join(data_path, i+'.wav'))))
    #             arr = np.array(dataset.get_register_data(os.path.join(data_path, i + '.wav')))
    #             li.append(model.predict(arr.reshape((1, *arr.shape))))
    #
    #         personLi = [person]
    #         personLi.extend((mean_vectors(li)[0]))
    #         groupLi = [name]
    #         groupLi.extend(personLi)
    #         rows_list.append(groupLi)
    # res = pd.DataFrame(rows_list)
    # print(res)
    # res.to_csv(os.path.join(root_file, 'enroll.csv'))
    # return rows_list





if __name__ == '__main__':
    test_output(if_handle_device=True)
