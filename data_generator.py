import numpy as np
from keras.utils import Sequence, to_categorical
from sklearn.utils import shuffle


class VoiceSequence(Sequence):

    def __init__(self, x_set, y_set, batch_size, num_class, data_process_func):
        self.x, self.y = shuffle(x_set, to_categorical(y_set, num_classes=num_class))
        self.batch_size = batch_size
        self.data_process_function = data_process_func
        self.length = self.__len__()

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        if idx == self.length - 2:
            self.__shuffle()
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return np.array(self.data_process_function(batch_x)), np.array(batch_y)

    def __shuffle(self):
        self.x, self.y = shuffle(self.x, self.y)
