import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout, TimeDistributed, Conv1D, GRU, BatchNormalization, Activation, LSTM, Reshape
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import classification_report

# Deactivate the GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# def my_weighted_loss(onehot_labels, logits):
#     """scale loss based on class weights
#     """
#     # compute weights based on their frequencies
#     class_weights = np.array([0.5, 5, 5]) # set your class weights here
#     # computer weights based on onehot labels
#     weights = tf.reduce_sum(class_weights * onehot_labels, axis=-1)
#     # compute (unweighted) softmax cross entropy loss
#     unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=[onehot_labels], logits=[logits])
#     # apply the weights, relying on broadcasting of the multiplication
#     weighted_losses = unweighted_losses * weights
#     # reduce the result to get your final loss
#     loss = tf.reduce_mean(weighted_losses)
#     return loss

# Clear back sessions
tf.keras.backend.clear_session()

# Some hyper-parameters
epochs = 500
batch_size = 256
filters = 64
kernel_size = 7
strides = 2
ego_dim = 6
object_slots_num = 6
obj_dim = 4
input_shape = (None, ego_dim+object_slots_num*obj_dim)
validation_split = 0.1
num_4_name = 21
learning_rate = 1e-1

path = r"..\preprocessed_data\test_with_steering_angle"
# path = r'..\preprocessed_data'
# path = r"..\preprocessed_data\test_without_steering_angle"

# Load the data
# X = np.load(r'{}\X.npy'.format(path))
# Y = np.load(r'{}\Y.npy'.format(path))
X = np.load('{}\\X_with_steering_with_lane_with_obj.npy'.format(path))
Y = np.load('{}\\Y_with_steering_with_lane_with_obj.npy'.format(path))

# Split the data into train and validation set
portion = int(X.shape[0] * validation_split)
X_validation = X[:portion, :, :]
Y_validation = Y[:portion, :, :]
X_train = X[portion:, :, :]
Y_train = Y[portion:, :, :]

model = Sequential([
    Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, input_shape=input_shape),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.5),
    GRU(256, return_sequences=True),
    Dropout(0.5),
    BatchNormalization(),
    GRU(256, return_sequences=True),
    Dropout(0.5),
    BatchNormalization(),
    Dropout(0.5),
    TimeDistributed(Dense(3, activation='softmax')),
])

model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['acc'])

history = model.fit(X_train, Y_train, batch_size=batch_size, validation_data=(X_validation, Y_validation), epochs=epochs, verbose=2)

model.save(r'{}\model_{}_new_lane_change_importance.h5'.format(path, num_4_name))

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
# plt.savefig(r"{}\training_loss_acc_{}".format(path, num_4_name))
plt.show()

# os.system("shutdown /s /t 1")

