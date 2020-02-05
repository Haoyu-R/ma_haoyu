import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, TimeDistributed, Conv1D, GRU, BatchNormalization, Activation, LSTM, Bidirectional
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
import os

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

suffixs = [0.5, 1.5, 2, 2.5, 3]

# Load the data
for suffix in suffixs:
    num_4_name = suffix

    X = np.load(r'{}\X_all_{}.npy'.format(path, suffix))
    Y = np.load(r'{}\Y_all_{}.npy'.format(path, suffix))

    # Split the data into train and validation set
    portion = int(X.shape[0] * validation_split)
    X_validation = X[:portion, :, :]
    Y_validation = Y[:portion, :, :]
    X_train = X[portion:, :, :]
    Y_train = Y[portion:, :, :]

    # Clear back sessions
    tf.keras.backend.clear_session()

    model = Sequential([
        Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, input_shape=input_shape),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),
        Bidirectional(LSTM(256, return_sequences=True)),
        Dropout(0.5),
        BatchNormalization(),
        Bidirectional(LSTM(256, return_sequences=True)),
        Dropout(0.5),
        BatchNormalization(),
        Dropout(0.5),
        TimeDistributed(Dense(5, activation='softmax'))
    ])

    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['acc'])

    history = model.fit(X_train, Y_train, batch_size=batch_size, validation_data=(X_validation, Y_validation), epochs=epochs, verbose=2)

    # Save the NN model
    model.save(r'{}\time_shift\model_{}.h5'.format(path, num_4_name))

    # Save the history dict
    hist_df = pd.DataFrame(history.history)
    hist_csv_file = r'{}\time_shift\history_{}.csv'.format(path, num_4_name)
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

    # Reshape Y to dim (timesteps*m, feature_dims)
    Y_valid_pred = model.predict(X_validation, verbose=2)
    Y_valid_pred = Y_valid_pred.reshape((Y_valid_pred.shape[0] * Y_valid_pred.shape[1], 5))
    Y_pred_bool = np.argmax(Y_valid_pred, axis=1)

    Y_validation = Y_validation.reshape((Y_validation.shape[0] * Y_validation.shape[1], 5))
    Y_valid_bool = np.argmax(Y_validation, axis=1)

    # Using sklearn to evaluate other metrics of model
    with open(r"{}\time_shift\report_{}.txt".format(path, num_4_name), "w") as text_file:
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
    plt.savefig(r"{}\time_shift\training_loss_acc_{}.png".format(path, num_4_name))
os.system("shutdown /s /t 1")
