import keras
from keras import Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.utils import plot_model

from resnet import resnet_v2
from transformer import get_transformer


def get_model(shape=(32, 1024), num_classes=500, model_type=0, **kwargs):
    if model_type == 0:
        return res_plus_transformer_model(shape, num_classes)
    elif model_type == 1:
        return simple_model(shape, num_classes)
    elif model_type == 2:
        return full_res_net_model(shape, num_classes, **kwargs)
    else:
        print("error")


def res_plus_transformer_model(shape=(32, 1024), num_classes=500):
    input_array = keras.Input(shape)

    three_d_input = keras.layers.Reshape(target_shape=(*shape, 1))(input_array)

    transformer_output = keras.layers.Flatten()(get_transformer(transformer_input=input_array, transformer_depth=3))
    resnet_output = resnet_v2(inputs=three_d_input, n=1)

    mid = keras.layers.concatenate([resnet_output, transformer_output])

    output = keras.layers.Dense(num_classes,
                                activation='relu',
                                kernel_initializer='he_normal')(mid)

    model = Model(inputs=input_array, outputs=output)
    model.compile(loss='categorical_crossentropy',
                  optimizer="sgd",
                  metrics=['accuracy'])
    model.summary()
    plot_model(model, to_file='model.png')
    return model


def simple_model(shape=(32, 1024), num_classes=500):
    model = Sequential()
    model.add(keras.layers.Reshape(target_shape=(*shape, 1), input_shape=shape))
    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model


def full_res_net_model(shape=(32, 1024), num_classes=500, n=1):
    input_array = keras.Input(shape)

    three_d_input = keras.layers.Reshape(target_shape=(*shape, 1))(input_array)
    resnet_output = resnet_v2(inputs=three_d_input, n=n)
    output = keras.layers.Dense(num_classes,
                                activation='softmax')(resnet_output)

    model = Model(inputs=input_array, outputs=output)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model.summary()
    return model


def load_model() -> keras.Model:
    """
    返回训练好的模型
    :return:
    """
    pass
