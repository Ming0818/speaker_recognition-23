import os
from typing import List

import keras
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
    model = load_model(model_path, model_type=1)
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


if __name__ == '__main__':
    y = get_mean_feature_for_device(model_path="./models/weights.04-6.36.hdf5", path="D:\\af2019-sr-devset-20190312",
                                    output_shape=(32, 1024), sample_rate=16000)
    print(y)
