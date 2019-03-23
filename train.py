import argparse

import keras
import keras.backend as K
import numpy as np
from keras import Input, Model
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Embedding, Lambda
from sklearn.model_selection import train_test_split

from dataset import DataSet
from loss import l2_softmax
from model import get_model, load_model

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
parser.add_argument('-lc', '--lambda_c', type=float, default=0.2, help='weight of center loss')
parser.add_argument('-l2', '--l2_lambda', type=int, default=10, help='lambda of l2-softmax')
parser.add_argument('--continue_training', action="store_true", help='if continue training by using model path')

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
feature_length = args.feature_length
lambda_c = args.lambda_c
l2_lambda = args.l2_lambda

print(args.continue_training)
# 保存模型!!!


# lr_scheduler = LearningRateScheduler(lr_schedule)
#
# lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
#                                cooldown=0,
#                                patience=5,
#                                min_lr=0.5e-6)

x, y = DataSet(file_dir=file_dir, output_shape=output_shape, sample_rate=sample_rate).get_train_data(
    process_class=process_class)
origin_y = np.array(y)
y = keras.utils.to_categorical(y, num_classes=class_num)
x, x_test, y, y_test, origin_y, origin_y_test = train_test_split(x, y, origin_y, test_size=0.25)
model = get_model(shape=output_shape, num_classes=class_num, model_type=model_type, n=net_depth,
                  feature_length=args.feature_length, l2_sm=l2_lambda, lambda_c=lambda_c)

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=16, write_graph=False, write_grads=False,
                          write_images=False, embeddings_freq=4, embeddings_layer_names='embedding_layer',
                          embeddings_data=[np.array(x[:class_num]),np.array(range(class_num))], update_freq='epoch')
# create dir
checkpoint = ModelCheckpoint(filepath='./models/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=False)

callbacks = [checkpoint]

if not args.continue_training and model_type == 4:
    x = np.array(x)
    x_test = np.array(x_test)
    random_y_train = np.random.rand(x.shape[0], 1)
    random_y_test = np.random.rand(x_test.shape[0], 1)
    model.fit(x=[x, np.array(origin_y)], y=[y, random_y_train], batch_size=batch_size, epochs=epochs,
              validation_data=([x_test, np.array(origin_y_test)], [y_test, random_y_test]), shuffle=True,
              callbacks=callbacks)
elif args.continue_training:
    # not able to run!
    print('continue')
    origin_model = load_model(model_path)
    origin_model.trainable = False
    origin_input = origin_model.input
    origin_feature_output = origin_model.get_layer('feature_layer').output
    origin_output = origin_model.output

    input_target = Input(shape=(1,))
    centers = Embedding(class_num, feature_length)(input_target)
    l2_loss = Lambda(lambda x: K.sum(K.square(x[0] - x[1][:, 0]), 1, keepdims=True), name='l2_loss')(
        [origin_feature_output, centers])

    model_center_loss = Model(inputs=[origin_input, input_target], outputs=[origin_output, l2_loss])
    model_center_loss.compile(optimizer=keras.optimizers.Adadelta(),
                              loss=[l2_softmax(l2_lambda), lambda y_true, y_pred: y_pred],
                              loss_weights=[1, lambda_c], metrics=['accuracy'])

    x = np.array(x)
    x_test = np.array(x_test)
    random_y_train = np.random.rand(x.shape[0], 1)
    random_y_test = np.random.rand(x_test.shape[0], 1)
    model.fit(x=[x, np.array(origin_y)], y=[y, random_y_train], batch_size=batch_size, epochs=1,
              validation_data=([x_test, np.array(origin_y_test)], [y_test, random_y_test]), shuffle=True,
              callbacks=callbacks)
    model.trainable = True
    model.fit(x=[x, np.array(origin_y)], y=[y, random_y_train], batch_size=batch_size, epochs=epochs-1,
              validation_data=([x_test, np.array(origin_y_test)], [y_test, random_y_test]), shuffle=True,
              callbacks=callbacks)

else:
    model.fit(np.array(x), y,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(np.array(x_test), y_test),
              shuffle=True,
              callbacks=callbacks
              )

if model_path is not None:
    model.save(model_path)
