import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, TimeDistributed, Conv1D, GRU, BatchNormalization, Activation
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

class Callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, logs={}):
        if logs.get('loss') < 0.4:
            print("Loss achieved, so cancel the progress")
            self.model.stop_training = True


tf.keras.backend.clear_session()

epochs = 100
batch_size = 64
filters = 64
kernel_size = 7
strides = 2
input_shape = (500, 4)

# lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10 ** (epoch/20))

X = np.load(r'..\preprocessed_data\test_with_steering_angle\X.npy')
Y = np.load(r'..\preprocessed_data\test_with_steering_angle\Y.npy')

print(X.shape)
print(Y.shape)
model = Sequential([
    Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, input_shape=input_shape, activation='relu'),
    # BatchNormalization(),
    # Activation('relu'),
    # Dropout(0.5),
    # GRU(128, return_sequences=True),
    GRU(128, return_sequences=True),
    TimeDistributed(Dense(3, activation='softmax'))
])

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[])
history = model.fit(X, Y, epochs=epochs, verbose=2)

# lrs = 1e-8 * (10 ** (epochs/20))
# plt.semilogx(lrs, history.history['loss'])
# plt.show()

model.save(r'..\preprocessed_data\MDM data process\NN_model\my_model.h5')