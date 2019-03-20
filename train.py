import argparse

import keras
import numpy as np
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

from dataset import DataSet
from model import get_model

parser = argparse.ArgumentParser("speaker recognition", fromfile_prefix_chars='@')
parser.add_argument('--file_dir', type=str, help='Directory to load data.')
parser.add_argument('--model_path', type=str, help='Directory to save model.')
parser.add_argument('-s', '--output_shape', type=int, nargs=2, default=[32, 1024], help='shape')
parser.add_argument('--hidden_size', type=int, default=64, help='The hidden full connected layer size')
parser.add_argument('-e', '--epochs_to_train', type=int, default=10, help='Number of epoch to train')
parser.add_argument('-b', '--batch_size', type=int, default=16, help='Number of training examples processed per step')
# parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='initial learning rate')
parser.add_argument('-sr', '--sample_rate', type=int, default=16000, help='sample rate of wave')
parser.add_argument('-c', '--class_num', type=int, default=103, help='class num of voice')
parser.add_argument('-pc', '--process_class', type=int, default=0, help='class of process\' way')
parser.add_argument('-mt', '--model_type', type=int, default=0,
                    help='type of model.0:res_plus_transformer; 1.simple_cnn; 2.res_net')
parser.add_argument('-n', '--net_depth', type=int, default=1,
                    help='net depth of res_net')
parser.add_argument('-fl', '--feature_length', type=int, default=200, help='feature length')

args = parser.parse_args()

model_path = args.model_path
batch_size = args.batch_size
sample_rate = args.sample_rate
epochs = args.epochs_to_train
file_dir = args.file_dir
output_shape = args.output_shape
class_num = args.class_num
process_class = args.process_class
model_type = args.model_type
net_depth = args.net_depth

# 保存模型!!!

checkpoint = ModelCheckpoint(filepath='./weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=False)

# lr_scheduler = LearningRateScheduler(lr_schedule)
#
# lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
#                                cooldown=0,
#                                patience=5,
#                                min_lr=0.5e-6)

x, y = DataSet(file_dir=file_dir, output_shape=output_shape, sample_rate=sample_rate).get_train_data(
    process_class=process_class)
y = keras.utils.to_categorical(y, num_classes=class_num)
x, x_test, y, y_test = train_test_split(x, y, test_size=0.25)
model = get_model(shape=output_shape, num_classes=class_num, model_type=model_type, n=net_depth, feature_length=args.feature_length)
callbacks = [checkpoint]

model.fit(np.array(x), y,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(np.array(x_test), y_test),
          shuffle=True,
          callbacks=callbacks
          )


if model_path is not None:
    model.save(model_path)

