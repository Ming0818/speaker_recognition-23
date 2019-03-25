import keras
import keras.backend as K
from keras import Model, Input
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Embedding, Lambda, BatchNormalization, GRU
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import plot_model

from loss import l2_softmax, bpr_triplet_loss, identity_loss
from resnet import resnet_v2
from transformer import get_transformer


def get_model(shape=(32, 1024), num_classes=500, model_type=0, **kwargs):
    if model_type == 0:
        return res_plus_transformer_model(shape, num_classes)
    elif model_type == 1:
        return simple_model(shape, num_classes)
    elif model_type == 2:
        return full_res_net_model(shape, num_classes, **kwargs)
    elif model_type == 3:
        return full_transformer(shape=shape, num_classes=num_classes)
    elif model_type == 4:
        return res_with_center_loss_model(shape, num_classes, **kwargs)
    elif model_type == 5:
        return triplet_loss(shape, num_classes, **kwargs)
    elif model_type == 6:
        return rnn(shape, num_classes, **kwargs)
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
    model.compile(loss=l2_softmax(5),
                  optimizer="sgd",
                  metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file='model.png')
    return model


def simple_model(shape=(32, 1024), num_classes=500, **kwargs):
    model = Sequential()
    model.add(keras.layers.Reshape(target_shape=(*shape, 1), input_shape=shape, name="input"))
    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', name="feature_layer"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax', name="output_layer"))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model


def rnn(shape=(32, 1024), num_classes=500, l2_sm=15, **kwargs):
    input_array = keras.Input(shape, name='input')
    rnn = GRU(50, return_sequences=False)(input_array)
    full = rnn
    den = Dense(50, activation="relu", name="feature_layer")(full)
    output = Dense(num_classes, activation='softmax', name="output_layer")(den)
    model = Model(inputs=input_array, outputs=output)
    model.compile(loss=l2_softmax(l2_sm),
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model.summary()
    return model


def full_res_net_model(shape=(32, 1024), num_classes=500, n=1, feature_length=100, l2_sm=15, **kwargs):
    input_array = keras.Input(shape, name='input')

    three_d_input = keras.layers.Reshape(target_shape=(*shape, 1))(input_array)
    resnet_output = resnet_v2(inputs=three_d_input, n=n)
    resnet_output = BatchNormalization()(resnet_output)
    mid = keras.layers.Dense(feature_length, activation='sigmoid', name="feature_layer")(resnet_output)
    # mid = Dropout(0.3)(mid)
    mid = BatchNormalization()(mid)
    output = keras.layers.Dense(num_classes, activation='softmax', name="output_layer")(mid)

    model = Model(inputs=input_array, outputs=output)
    model.compile(loss=l2_softmax(l2_sm),
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model.summary()
    return model


def triplet_loss(shape=(32, 1024), num_classes=500, n=1, feature_length=100, l2_sm=15, **kwargs):
    origin_model = full_res_net_model(shape, num_classes, n, feature_length, l2_sm)
    origin_input = origin_model.input
    origin_feature_output = origin_model.get_layer('feature_layer').output
    origin_output = origin_model.get_layer('output_layer').output

    model = Model(inputs=origin_input, outputs=origin_feature_output)

    user_input = Input(shape, name='user_input')
    positive_item_input = Input(shape, name='positive_item_input')
    negative_item_input = Input(shape, name='negative_item_input')

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
    plot_model(model, to_file='model.png')
    return model


def res_with_center_loss_model(shape=(32, 1024), num_classes=500, n=1, feature_length=100, l2_sm=15, lambda_c=0.2):
    origin_model = full_res_net_model(shape, num_classes, n, feature_length, l2_sm)
    origin_input = origin_model.input
    origin_feature_output = origin_model.get_layer('feature_layer').output
    origin_output = origin_model.get_layer('output_layer').output

    input_target = Input(shape=(1,))
    centers = Embedding(num_classes, feature_length, name='embedding_layer')(input_target)
    l2_loss = Lambda(lambda x: K.sum(K.square(x[0] - x[1][:, 0]), 1, keepdims=True), name='l2_loss')(
        [origin_feature_output, centers])

    model_center_loss = Model(inputs=[origin_input, input_target], outputs=[origin_output, l2_loss])
    model_center_loss.compile(optimizer=keras.optimizers.Adadelta(),
                              loss=[l2_softmax(l2_sm), lambda y_true, y_pred: y_pred],
                              loss_weights=[1, lambda_c], metrics=['accuracy'])
    return model_center_loss


def full_transformer(shape=(32, 1024), num_classes=500, feature_length=100):
    input_array = keras.Input(shape)

    transformer_output = keras.layers.Flatten()(get_transformer(transformer_input=input_array, transformer_depth=3))
    btchn = BatchNormalization()(transformer_output)
    mid = keras.layers.Dense(feature_length, activation='relu', name="feature_layer")(btchn)

    output = keras.layers.Dense(num_classes,
                                activation='sigmoid',
                                kernel_initializer='he_normal')(mid)

    model = Model(inputs=input_array, outputs=output)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model.summary()
    return model


def load_model(model_path, model_type=0) -> keras.Model:
    """
    返回训练好的模型
    :return:
    """

    def temp(a, b):
        return b

    if model_type == 0:
        return keras.models.load_model(model_path, custom_objects={'internal': l2_softmax(10), '<lambda>': temp})
    elif model_type == 2:
        model = keras.models.load_model(model_path, custom_objects={'internal': l2_softmax(10), '<lambda>': temp})
        output = model.get_layer('feature_layer').output
        new_model = Model(inputs=model.get_layer('input').input, outputs=output)
        return new_model
    elif model_type == 1:
        model = keras.models.load_model(model_path, custom_objects={'internal': l2_softmax(10)})
        output = model.get_layer('feature_layer').output
        new_model = Model(inputs=model.input, outputs=output)
        return new_model
