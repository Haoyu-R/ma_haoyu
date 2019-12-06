import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, TimeDistributed, Conv1D, GRU, BatchNormalization, Activation
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import classification_report


# Deactivate the GPU
os.environ["CUDA_VISIBLE_DEVICES"]="-1"


# Callback to early stop training
class Callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, logs={}):
        if logs.get('loss') < 0.4:
            print("Loss achieved, so cancel the progress")
            self.model.stop_training = True


# Clear back sessions
tf.keras.backend.clear_session()


# Some hyper-parameters
epochs = 500
batch_size = 64
filters = 64
kernel_size = 7
strides = 2
input_shape = (500, 4)
validation_split = 0.3

# Used to select the best learning rate
# lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10 ** (epoch/20))

X = np.load(r'..\preprocessed_data\test_with_steering_angle\X.npy')
Y = np.load(r'..\preprocessed_data\test_with_steering_angle\Y.npy')

# Split the data into train and validation set
portion = int(X.shape[0]*validation_split)
X_validation = X[:portion, :, :]
Y_validation = Y[:portion, :, :]
X_train = X[portion:, :, :]
Y_train = Y[portion:, :, :]
print("Train on {} samples".format(X.shape[0]-portion))
print("Validate on {} samples".format(portion))

model = Sequential([
    Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, input_shape=input_shape),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.5),
    GRU(128, return_sequences=True),
    Dropout(0.5),
    BatchNormalization(),
    GRU(128, return_sequences=True),
    TimeDistributed(Dense(3, activation='softmax'))
])

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
history = model.fit(X_train, Y_train, validation_data=(X_validation, Y_validation), epochs=epochs, verbose=2)


# # Used to find best learning rate
# lrs = 1e-8 * (10 ** (epochs/20))
# plt.semilogx(lrs, history.history['loss'])
# plt.show()


# Reshape Y to dim (timesteps*m, feature_dims)
Y_valid_pred = model.predict(X_validation, verbose=2)
Y_valid_pred = Y_valid_pred.reshape((Y_valid_pred.shape[0]*Y_valid_pred.shape[1], 3))
Y_pred_bool = np.argmax(Y_valid_pred, axis=1)

Y_validation = Y_validation.reshape((Y_validation.shape[0]*Y_validation.shape[1], 3))
Y_valid_bool = np.argmax(Y_validation, axis=1)

# Using sklearn to evaluate other metrics of model
print(classification_report(Y_valid_bool, Y_pred_bool))

# Plot history for accuracy
plt.subplot(2, 1, 1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

# Plot history for loss
plt.subplot(2, 1, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Following visualize the label in on sample
y_test = model.predict(np.expand_dims(X[0, :, :], axis=0))
plt.subplot(2, 3, 1)
plt.plot(y_test[0, :, 0])
plt.title('Free driving label - Predicted ')

plt.subplot(2, 3, 2)
plt.plot(y_test[0, :, 1])
plt.title('Left lane change label - Predicted ')

plt.subplot(2, 3, 3)
plt.plot(y_test[0, :, 2])
plt.title('Right lane change label - Predicted ')

plt.subplot(2, 3, 4)
plt.plot(Y[0, :, 0])
plt.title('Free driving label - Real')

plt.subplot(2, 3, 5)
plt.plot(Y[0, :, 1])
plt.title('Left lane change label - Real')

plt.subplot(2, 3, 6)
plt.plot(Y[0, :, 2])
plt.title('Right lane change label - Real')
plt.show()

# Save the NN model
# model.save(r'..\preprocessed_data\test_with_steering_angle\my_model.h5')