import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, TimeDistributed, Conv1D, GRU, BatchNormalization, Activation, LSTM, Bidirectional
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
import os


def my_weighted_loss(onehot_labels, logits):
    """scale loss based on class weights
    """
    # compute weights based on their frequencies
    class_weights = np.array([0.24, 1.41, 1.41, 17.4, 17.4]) # set your class weights here
    # computer weights based on onehot labels
    weights = tf.reduce_sum(class_weights * onehot_labels, axis=-1)
    # compute (unweighted) softmax cross entropy loss
    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=[onehot_labels], logits=[logits])
    # apply the weights, relying on broadcasting of the multiplication
    weighted_losses = unweighted_losses * weights
    # reduce the result to get your final loss
    loss = tf.reduce_mean(weighted_losses)
    return loss

# Some hyper-parameters
epochs = 500
batch_size = 128
filters = 64
kernel_size = 7
strides = 2
ego_dim = 4
object_slots_num = 6
obj_dim = 4
input_shape = (None, ego_dim+object_slots_num*obj_dim)
validation_split = 0.1
learning_rate = 1e-1
path = r"..\preprocessed_data\test_with_steering_angle"


# Load the data
X = np.load(r'{}\X_all.npy'.format(path))
Y = np.load(r'{}\Y_all.npy'.format(path))

# Split the data into train and validation set
portion = int(X.shape[0] * validation_split)
X_validation = X[:portion, :, :]
Y_validation = Y[:portion, :, :]
X_train = X[portion:, :, :]
Y_train = Y[portion:, :, :]

###############################################################################################

# num_4_name = 1
#
# # Clear back sessions
# tf.keras.backend.clear_session()
#
# model = Sequential([
#     Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, input_shape=input_shape),
#     BatchNormalization(),
#     Activation('relu'),
#     Dropout(0.5),
#     GRU(256, return_sequences=True),
#     Dropout(0.5),
#     BatchNormalization(),
#     GRU(256, return_sequences=True),
#     Dropout(0.5),
#     BatchNormalization(),
#     Dropout(0.5),
#     TimeDistributed(Dense(5, activation='softmax'))
# ])
#
# model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['acc'])
#
# history = model.fit(X_train, Y_train, batch_size=batch_size, validation_data=(X_validation, Y_validation), epochs=epochs, verbose=2)
#
# # Save the NN model
# model.save(r'{}\final\model_{}.h5'.format(path, num_4_name))
#
# # Save the history dict
# hist_df = pd.DataFrame(history.history)
# hist_csv_file = r'{}\final\history_{}.csv'.format(path, num_4_name)
# with open(hist_csv_file, mode='w') as f:
#     hist_df.to_csv(f)
#
# # Reshape Y to dim (timesteps*m, feature_dims)
# Y_valid_pred = model.predict(X_validation, verbose=2)
# Y_valid_pred = Y_valid_pred.reshape((Y_valid_pred.shape[0] * Y_valid_pred.shape[1], 5))
# Y_pred_bool = np.argmax(Y_valid_pred, axis=1)
#
# Y_validation = Y_validation.reshape((Y_validation.shape[0] * Y_validation.shape[1], 5))
# Y_valid_bool = np.argmax(Y_validation, axis=1)
#
# # Using sklearn to evaluate other metrics of model
# with open(r"{}\final\report_{}.txt".format(path, num_4_name), "w") as text_file:
#     text_file.write(classification_report(Y_valid_bool, Y_pred_bool))
#
# # Plot history for accuracy
# plt.subplot(2, 1, 1)
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper right')
#
# # Plot history for loss
# plt.subplot(2, 1, 2)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper right')
# plt.tight_layout()
#
# # Save the training history picture
# plt.savefig(r"{}\final\training_loss_acc_{}".format(path, num_4_name))


###############################################################################################
# num_4_name = 2
#
# # Clear back sessions
# tf.keras.backend.clear_session()
#
# model = Sequential([
#     Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, input_shape=input_shape),
#     BatchNormalization(),
#     Activation('relu'),
#     Dropout(0.5),
#     LSTM(256, return_sequences=True),
#     Dropout(0.5),
#     BatchNormalization(),
#     LSTM(256, return_sequences=True),
#     Dropout(0.5),
#     BatchNormalization(),
#     Dropout(0.5),
#     TimeDistributed(Dense(5, activation='softmax'))
# ])
#
# model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['acc'])
#
# history = model.fit(X_train, Y_train, batch_size=batch_size, validation_data=(X_validation, Y_validation), epochs=epochs, verbose=2)
#
# # Save the NN model
# model.save(r'{}\final\model_{}.h5'.format(path, num_4_name))
#
# # Save the history dict
# hist_df = pd.DataFrame(history.history)
# hist_csv_file = r'{}\final\history_{}.csv'.format(path, num_4_name)
# with open(hist_csv_file, mode='w') as f:
#     hist_df.to_csv(f)
#
# # Reshape Y to dim (timesteps*m, feature_dims)
# Y_valid_pred = model.predict(X_validation, verbose=2)
# Y_valid_pred = Y_valid_pred.reshape((Y_valid_pred.shape[0] * Y_valid_pred.shape[1], 5))
# Y_pred_bool = np.argmax(Y_valid_pred, axis=1)
#
# Y_validation = Y_validation.reshape((Y_validation.shape[0] * Y_validation.shape[1], 5))
# Y_valid_bool = np.argmax(Y_validation, axis=1)
#
# # Using sklearn to evaluate other metrics of model
# with open(r"{}\final\report_{}.txt".format(path, num_4_name), "w") as text_file:
#     text_file.write(classification_report(Y_valid_bool, Y_pred_bool))
#
# # Plot history for accuracy
# plt.subplot(2, 1, 1)
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper right')
#
# # Plot history for loss
# plt.subplot(2, 1, 2)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper right')
# plt.tight_layout()
#
# # Save the training history picture
# plt.savefig(r"{}\final\training_loss_acc_{}".format(path, num_4_name))

#
# ###############################################################################################
#
# num_4_name = 3
#
# # Clear back sessions
# tf.keras.backend.clear_session()
#
# model = Sequential([
#     Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, input_shape=input_shape),
#     BatchNormalization(),
#     Activation('relu'),
#     Dropout(0.5),
#     Bidirectional(GRU(256, return_sequences=True)),
#     Dropout(0.5),
#     BatchNormalization(),
#     Bidirectional(GRU(256, return_sequences=True)),
#     Dropout(0.5),
#     BatchNormalization(),
#     Dropout(0.5),
#     TimeDistributed(Dense(5, activation='softmax'))
# ])
#
# model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['acc'])
#
# history = model.fit(X_train, Y_train, batch_size=batch_size, validation_data=(X_validation, Y_validation), epochs=epochs, verbose=2)
#
# # Save the NN model
# model.save(r'{}\final\model_{}.h5'.format(path, num_4_name))
#
# # Save the history dict
# hist_df = pd.DataFrame(history.history)
# hist_csv_file = r'{}\final\history_{}.csv'.format(path, num_4_name)
# with open(hist_csv_file, mode='w') as f:
#     hist_df.to_csv(f)
#
# # Reshape Y to dim (timesteps*m, feature_dims)
# Y_valid_pred = model.predict(X_validation, verbose=2)
# Y_valid_pred = Y_valid_pred.reshape((Y_valid_pred.shape[0] * Y_valid_pred.shape[1], 5))
# Y_pred_bool = np.argmax(Y_valid_pred, axis=1)
#
# Y_validation = Y_validation.reshape((Y_validation.shape[0] * Y_validation.shape[1], 5))
# Y_valid_bool = np.argmax(Y_validation, axis=1)
#
# # Using sklearn to evaluate other metrics of model
# with open(r"{}\final\report_{}.txt".format(path, num_4_name), "w") as text_file:
#     text_file.write(classification_report(Y_valid_bool, Y_pred_bool))
#
# # Plot history for accuracy
# plt.subplot(2, 1, 1)
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper right')
#
# # Plot history for loss
# plt.subplot(2, 1, 2)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper right')
# plt.tight_layout()
#
# # Save the training history picture
# plt.savefig(r"{}\final\training_loss_acc_{}".format(path, num_4_name))
#
#
# ###############################################################################################
#
# num_4_name = 4
#
# # Clear back sessions
# # tf.keras.backend.clear_session()
#
# model = Sequential([
#     Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, input_shape=input_shape),
#     BatchNormalization(),
#     Activation('relu'),
#     Dropout(0.5),
#     Bidirectional(LSTM(256, return_sequences=True)),
#     Dropout(0.5),
#     BatchNormalization(),
#     Bidirectional(LSTM(256, return_sequences=True)),
#     Dropout(0.5),
#     BatchNormalization(),
#     Dropout(0.5),
#     TimeDistributed(Dense(5, activation='softmax'))
# ])
#
# model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['acc'])
#
# history = model.fit(X_train, Y_train, batch_size=batch_size, validation_data=(X_validation, Y_validation), epochs=epochs, verbose=2)
#
# # Save the NN model
# model.save(r'{}\final\model_{}.h5'.format(path, num_4_name))
#
# # Save the history dict
# hist_df = pd.DataFrame(history.history)
# hist_csv_file = r'{}\final\history_{}.csv'.format(path, num_4_name)
# with open(hist_csv_file, mode='w') as f:
#     hist_df.to_csv(f)
#
# # Reshape Y to dim (timesteps*m, feature_dims)
# Y_valid_pred = model.predict(X_validation, verbose=2)
# Y_valid_pred = Y_valid_pred.reshape((Y_valid_pred.shape[0] * Y_valid_pred.shape[1], 5))
# Y_pred_bool = np.argmax(Y_valid_pred, axis=1)
#
# Y_validation = Y_validation.reshape((Y_validation.shape[0] * Y_validation.shape[1], 5))
# Y_valid_bool = np.argmax(Y_validation, axis=1)
#
# # Using sklearn to evaluate other metrics of model
# with open(r"{}\final\report_{}.txt".format(path, num_4_name), "w") as text_file:
#     text_file.write(classification_report(Y_valid_bool, Y_pred_bool))
#
# # Plot history for accuracy
# plt.subplot(2, 1, 1)
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper right')
#
# # Plot history for loss
# plt.subplot(2, 1, 2)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper right')
# plt.tight_layout()
#
# # Save the training history picture
# plt.savefig(r"{}\final\training_loss_acc_{}".format(path, num_4_name))


# ###############################################################################################
#
num_4_name = 5
batch_size = 256
# Clear back sessions
# tf.keras.backend.clear_session()

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
    TimeDistributed(Dense(5, activation='softmax'))
])

model.compile(optimizer="adam", loss=my_weighted_loss, metrics=['acc'])

history = model.fit(X_train, Y_train, batch_size=batch_size, validation_data=(X_validation, Y_validation), epochs=epochs, verbose=2)

# Save the NN model
model.save(r'{}\final\model_{}.h5'.format(path, num_4_name))

# Save the history dict
hist_df = pd.DataFrame(history.history)
hist_csv_file = r'{}\final\history_{}.csv'.format(path, num_4_name)
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

# Reshape Y to dim (timesteps*m, feature_dims)
Y_valid_pred = model.predict(X_validation, verbose=2)
Y_valid_pred = Y_valid_pred.reshape((Y_valid_pred.shape[0] * Y_valid_pred.shape[1], 5))
Y_pred_bool = np.argmax(Y_valid_pred, axis=1)

Y_validation = Y_validation.reshape((Y_validation.shape[0] * Y_validation.shape[1], 5))
Y_valid_bool = np.argmax(Y_validation, axis=1)

# Using sklearn to evaluate other metrics of model
with open(r"{}\final\report_{}.txt".format(path, num_4_name), "w") as text_file:
    text_file.write(classification_report(Y_valid_bool, Y_pred_bool))

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
plt.savefig(r"{}\final\training_loss_acc_{}".format(path, num_4_name))


###############################################################################################

# num_4_name = 6
# batch_size = 128
# # Clear back sessions
# # tf.keras.backend.clear_session()
#
# model = Sequential([
#     Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, input_shape=input_shape),
#     BatchNormalization(),
#     Activation('relu'),
#     Dropout(0.5),
#     Bidirectional(GRU(256, return_sequences=True)),
#     Dropout(0.5),
#     BatchNormalization(),
#     Bidirectional(GRU(256, return_sequences=True)),
#     Dropout(0.5),
#     BatchNormalization(),
#     Dropout(0.5),
#     TimeDistributed(Dense(5, activation='softmax'))
# ])
#
# model.compile(optimizer="adam", loss=my_weighted_loss, metrics=['acc'])
#
# history = model.fit(X_train, Y_train, batch_size=batch_size, validation_data=(X_validation, Y_validation), epochs=epochs, verbose=2)
#
# # Save the NN model
# model.save(r'{}\final\model_{}.h5'.format(path, num_4_name))
#
# # Save the history dict
# hist_df = pd.DataFrame(history.history)
# hist_csv_file = r'{}\final\history_{}.csv'.format(path, num_4_name)
# with open(hist_csv_file, mode='w') as f:
#     hist_df.to_csv(f)
#
# # Reshape Y to dim (timesteps*m, feature_dims)
# Y_valid_pred = model.predict(X_validation, verbose=2)
# Y_valid_pred = Y_valid_pred.reshape((Y_valid_pred.shape[0] * Y_valid_pred.shape[1], 5))
# Y_pred_bool = np.argmax(Y_valid_pred, axis=1)
#
# Y_validation = Y_validation.reshape((Y_validation.shape[0] * Y_validation.shape[1], 5))
# Y_valid_bool = np.argmax(Y_validation, axis=1)
#
# # Using sklearn to evaluate other metrics of model
# with open(r"{}\final\report_{}.txt".format(path, num_4_name), "w") as text_file:
#     text_file.write(classification_report(Y_valid_bool, Y_pred_bool))
#
# # Plot history for accuracy
# plt.subplot(2, 1, 1)
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper right')
#
# # Plot history for loss
# plt.subplot(2, 1, 2)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper right')
# plt.tight_layout()
#
# # Save the training history picture
# plt.savefig(r"{}\final\training_loss_acc_{}".format(path, num_4_name))
#
# os.system("shutdown /s /t 1")


