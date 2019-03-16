import numpy as np
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

from dataset import DataSet
from model import get_model
from resnet import lr_schedule

filepath = "./"
batch_size = 16
epochs = 100

checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

x, y = DataSet(file_dir="").get_train_data()
x, x_test, y, y_test = train_test_split(x, y, test_size=0.25)
model = get_model()
callbacks = [checkpoint, lr_reducer, lr_scheduler]

model.fit(x, y,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True,
          callbacks=callbacks)
