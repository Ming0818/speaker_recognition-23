from typing import List

import keras
import numpy as np
import scipy


def get_vector(wav_file_list, model: keras.Model):
    """

    :param wav_file_list: wav file should have been proccessed before input
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
    if dis_type == 0:
        # dist = 1.0 - uv / np.sqrt(uu * vv)
        for i in range(vectors.shape[0]):
            result.append(scipy.spatial.distance.cosine(anchor_vector, vectors[i]))
        return result


def mean_vectors(vectors):
    return np.mean(vectors , axis=0)


if __name__ == '__main__':
    print(distance(np.ones(shape=(10,)), np.ones(shape=(10, 10))))
