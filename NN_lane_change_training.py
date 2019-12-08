import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, TimeDistributed, Conv1D, GRU, BatchNormalization, Activation
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
from keras.utils import plot_model
import pandas as pd
from sklearn.metrics import classification_report
from keras.callbacks import TensorBoard

# Deactivate the GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def lr_schedule(epoch):
    if epoch < 5:
        return 0.1
    if 5 <= epoch < 300:
        return 0.001
    if 300 <= epoch < 500:
        return 0.0001
    if 500 <= epoch < 700:
        return 0.00001
    if 700 <= epoch < 900:
        return 0.000001
    if 900 <= epoch:
        return 0.0000001


# Callback to early stop training
class Callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, logs={}):
        if logs.get('loss') < 0.4:
            print("Loss achieved, so cancel the progress")
            self.model.stop_training = True


# Clear back sessions
tf.keras.backend.clear_session()

# Some hyper-parameters
epochs = 1000
batch_size = 64
filters = 64
kernel_size = 7
strides = 2
input_shape = (None, 4)
validation_split = 0.15
num_4_name = 3
learning_rate = 1e-1
path = r"..\preprocessed_data\test_with_steering_angle"

# Used to select the best learning rate
# lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10 ** (epoch/20))

# Load the data
X = np.load(r'{}\X.npy'.format(path))
Y = np.load(r'{}\Y.npy'.format(path))

# Split the data into train and validation set
portion = int(X.shape[0] * validation_split)
X_validation = X[:portion, :, :]
Y_validation = Y[:portion, :, :]
X_train = X[portion:, :, :]
Y_train = Y[portion:, :, :]
# print("Train on {} samples".format(X.shape[0]-portion))
# print("Validate on {} samples".format(portion))

model = Sequential([
    Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, input_shape=input_shape),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.5),
    GRU(128, return_sequences=True),
    Dropout(0.5),
    BatchNormalization(),
    GRU(128, return_sequences=True),
    Dropout(0.5),
    BatchNormalization(),
    Dropout(0.5),
    TimeDistributed(Dense(3, activation='softmax'))
])

# #print the computation graph in png
# plot_model(model, to_file=r'{}\model_all_{}.png'.format(path, num_4_name), show_shapes=True, show_layer_names=True)


model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['acc'])

log_dir = "logs\\fit{}\\".format(num_4_name) + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

history = model.fit(X_train, Y_train, validation_data=(X_validation, Y_validation), epochs=epochs, verbose=2, callbacks=[tensorboard_callback, lr_scheduler])

# Used to find best learning rate
# lrs = learning_rate * (10 ** (np.arange(epochs)/20))
# plt.semilogx(lrs, history.history['loss'])
# plt.show()

# Save the NN model
model.save(r'{}\model_all_{}.h5'.format(path, num_4_name))

# Reshape Y to dim (timesteps*m, feature_dims)
Y_valid_pred = model.predict(X_validation, verbose=2)
Y_valid_pred = Y_valid_pred.reshape((Y_valid_pred.shape[0] * Y_valid_pred.shape[1], 3))
Y_pred_bool = np.argmax(Y_valid_pred, axis=1)

Y_validation = Y_validation.reshape((Y_validation.shape[0] * Y_validation.shape[1], 3))
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
plt.legend(['train', 'test'], loc='upper right')

# Plot history for loss
plt.subplot(2, 1, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.tight_layout()
# Save the training history picture
plt.savefig(r"{}\training_loss_acc_{}".format(path, num_4_name))
# plt.show()


# Save the history dict
hist_df = pd.DataFrame(history.history)
hist_csv_file = r'{}\history_{}.csv'.format(path, num_4_name)
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

os.system("shutdown /s /t 1")
