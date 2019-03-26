import argparse

import keras
import numpy as np
from keras import Model, Input
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Lambda
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from dataset import DataSet
from loss import l2_softmax, bpr_triplet_loss, identity_loss
from model import load_model

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

# x, y = DataSet(file_dir=file_dir, output_shape=output_shape, sample_rate=sample_rate).get_train_data(
#     process_class=process_class)
# print("read")
# origin_y = np.array(y)
# y = keras.utils.to_categorical(y, num_classes=class_num)
# x, x_test, y, y_test, origin_y, origin_y_test = train_test_split(x, y, origin_y, test_size=0.25)

# tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=16, write_graph=False, write_grads=False,
#                           write_images=False, embeddings_freq=4, embeddings_layer_names='embedding_layer',
#                           embeddings_data=[np.array(x[:class_num]), np.array(range(class_num))], update_freq='epoch')
# create dir
checkpoint = ModelCheckpoint(filepath='./models/weights.{epoch:02d}-{loss:.2f}.hdf5',
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=False)

model = load_model(model_path, load_type=0)


origin_input = model.input
origin_feature_output = model.get_layer('feature_layer').output
origin_output = model.get_layer('output_layer').output

model = Model(inputs=origin_input, outputs=origin_feature_output)

user_input = Input(output_shape, name='user_input')
positive_item_input = Input(output_shape, name='positive_item_input')
negative_item_input = Input(output_shape, name='negative_item_input')

user_output = model(user_input)
positive_item_output = model(positive_item_input)
negative_item_output = model(negative_item_input)

loss = Lambda(
    lambda x: bpr_triplet_loss(x),
    name='loss',
    output_shape=(1,))([positive_item_output, negative_item_output, user_output])

model = Model(
    input=[positive_item_input, negative_item_input, user_input],
    output=loss)
model.compile(loss=identity_loss, optimizer=Adam())
model.summary()

ds = DataSet(file_dir=file_dir, output_shape=output_shape, sample_rate=sample_rate, batch_size=5000)

model.fit_generator(ds.get_triplet_batch(8), epochs=10, steps_per_epoch=1000, callbacks=[checkpoint])
