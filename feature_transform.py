import os
from typing import List

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from dataset import DataSet
from model import load_model


def get_vector(wav_file_list, model: keras.Model):
    """

    :param wav_file_list: wav file should have been processed before input
    :param model:
    :return:
    """
    return model.predict(np.array(wav_file_list))


def distance(anchor_vector: np.array, vectors: np.array, dis_type=0) -> List[float]:
    """
    warning! No Zeros
    :param anchor_vector: shape(feature_length, )
    :param vectors: shape(num_of_voice, feature_length)
    :param dis_type:
    :return:
    """

    result = []
    # dist = 1.0 - uv / np.sqrt(uu * vv)
    for i in range(vectors.shape[0]):
        if dis_type == 0:
            result.append(scipy.spatial.distance.cosine(anchor_vector, vectors[i]))
        elif dis_type == 1:
            result.append(scipy.spatial.distance.euclidean(anchor_vector, vectors[i]))
    return result


def mean_vectors(vectors):
    return np.mean(vectors, axis=0)


def get_mean_feature_for_device(path, model_path, output_shape, sample_rate, process_class=1):
    model = load_model(model_path, model_type=2)
    data = pd.read_csv(os.path.join(path, "enrollment.csv"))
    dataset = DataSet(file_dir=path, output_shape=output_shape, sample_rate=sample_rate)
    feature_dict = {}
    for device_id, df in data.groupby('DeviceID'):
        file_name = df['FileID']
        feature_dict[device_id] = []
        for i in file_name:
            feature_dict[device_id].append(
                np.array(dataset.get_register_data(os.path.join(path, "data", i + '.wav'), process_class)))

    for i in feature_dict.keys():
        feature_dict[i] = mean_vectors(model.predict(np.array(feature_dict[i])))

    return feature_dict


def get_threshold(model_path, path, output_shape, sample_rate):
    files, labels = DataSet(file_dir=path, output_shape=output_shape, sample_rate=sample_rate).get_train_data(
        process_class=1)
    model = load_model(model_path, model_type=2)

    f = []
    target_label = 8
    for file, label in zip(files, labels):
        if label == target_label:
            f.append(file)

    features = model.predict(np.array(f))

    mean = mean_vectors(features)
    dis = distance(mean, features)
    k = pd.Series(dis)
    k.hist()
    plt.show()


def device_test():
    ds = DataSet(file_dir='D:\\af2019-sr-devset-20190312\\data', output_shape=(32, 1024), sample_rate=16000)
    model = load_model("weights.05-4.37.hdf5", model_type=2)
    a = ds._process_data(ds._read_data('c2574ce9bf4a9b20e945f98c5dd4d40c.wav'), process_class=1)
    b = ds._process_data(ds._read_data('04553c11a57122b9b05ac7b534285bf8.wav'), process_class=1)

    result = model.predict(np.array([a, b]))
    k = distance(result[0], result[1].reshape((1, -1)))
    print(k)
    mean = get_mean_feature_for_device(model_path="./models/weights.04-6.36.hdf5", path="D:\\af2019-sr-devset-20190312",
                                output_shape=(32, 1024), sample_rate=16000)
    result[0] = np.subtract(result[0], mean[0])
    result[1] = np.subtract(result[1], mean[1])
    k = distance(result[0], result[1].reshape((1, -1)))
    print(k)


if __name__ == '__main__':
    device_test()
    # get_threshold("weights.05-4.37.hdf5", "D:\\2019af-sr-aishell2\\AISHELL-2\\iOS\\data\\data", (32, 1024), 16000)
    # print(distance(np.ones(shape=(10,)), np.ones(shape=(10, 10))))
    # y = get_mean_feature_for_device(model_path="./models/weights.04-6.36.hdf5", path="D:\\af2019-sr-devset-20190312",
    #                                 output_shape=(32, 1024), sample_rate=16000)
    # print(y)
