import keras
from keras import Model
from keras.optimizers import Adam
from keras.utils import plot_model
from resnet import resnet_v2, lr_schedule
from transformer import get_transformer

num_classes = 2000
shape = (24, 1024)
input_array = keras.Input(shape)

three_d_input = keras.layers.Reshape(target_shape=(*shape, 1))(input_array)

transformer_output = keras.layers.Flatten()(get_transformer(transformer_input=input_array, transformer_depth=3))
resnet_output = resnet_v2(inputs=three_d_input, n=1)

mid = keras.layers.concatenate([resnet_output, transformer_output])

output = keras.layers.Dense(num_classes,
                            activation='softmax',
                            kernel_initializer='he_normal')(mid)

model = Model(inputs=input_array, outputs=output)
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])
model.summary()
plot_model(model, to_file='model.png')
