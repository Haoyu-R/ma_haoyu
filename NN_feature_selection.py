import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, TimeDistributed, Conv1D, GRU, BatchNormalization, Activation
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
import eli5
from eli5.sklearn import PermutationImportance


# Deactivate the GPU
os.environ["CUDA_VISIBLE_DEVICES"]="-1"


def lane_change_model(filters_=64, kernel_size_=7, strides_=2, input_shape_=(500, 4)):

    model = Sequential()
    model.add(Conv1D(filters=filters_, kernel_size=kernel_size_, strides=strides_, input_shape=input_shape_))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(GRU(128, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(GRU(128, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(3, activation='softmax')))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    return model


# Clear back sessions
tf.keras.backend.clear_session()


# Some hyper-parameters
epochs = 2
batch_size = 64
filters = 64
kernel_size = 7
strides = 2
input_shape = (500, 4)
validation_split = 0.3

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

model = lane_change_model()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
history = model.fit(X_train, Y_train, validation_data=(X_validation, Y_validation), epochs=epochs, verbose=2)

# Feature selection
# feature_model = KerasRegressor(build_fn=lane_change_model)
# feature_model.fit(X_train, Y_train)
# Y_validation = Y_validation.reshape((Y_validation.shape[0]*Y_validation.shape[1], 3))
# Y_valid_bool = np.argmax(Y_validation, axis=1)
# X_validation = X_validation.reshape((X_validation.shape[0]*X_validation.shape[1], 4))


# perm = PermutationImportance(model , random_state=1, scoring="accuracy").fit(X_validation, Y_valid_bool)
# eli5.show_weights(perm, feature_names = X.columns.tolist())


