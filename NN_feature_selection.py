import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, TimeDistributed, Conv1D, GRU, BatchNormalization, Activation
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import classification_report
from keras.models import load_model
import random
from keras.losses import CategoricalCrossentropy

# Deactivate the GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Clear back sessions
tf.keras.backend.clear_session()


# Some hyper-parameters
validation_split = 0.1

X = np.load(r'NN_data\X.npy')
Y = np.load(r'NN_data\Y.npy')

# Split the data into train and validation set
portion = int(X.shape[0]*validation_split)
X_validation = X[:portion, :, 1:]
Y_validation = Y[:portion, :, :]
X_train = X[portion:, :, 1:]
Y_train = Y[portion:, :, :]
# print("Train on {} samples".format(X.shape[0]-portion))
# print("Validate on {} samples".format(portion))
#
# model = lane_change_model()
#
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
# model.fit(X_train, Y_train, epochs=epochs, verbose=2, validation_data=(X_validation, Y_validation))

# model.save(r'NN_data\my_model.h5')

# # Feature selection
trained_model = load_model(r'NN_data\model_without_steering_ang_6.h5')

# Reshape predicted Y to dim (timesteps*m, feature_dims)
Y_valid_pred = trained_model.predict(X_validation, verbose=2)
Y_valid_pred_temp = Y_valid_pred.reshape((Y_valid_pred.shape[0]*Y_valid_pred.shape[1], 3))
Y_valid_pred_bool = np.argmax(Y_valid_pred_temp, axis=1)

# Reshape real Y to dim
Y_validation_temp = Y_validation.reshape((Y_validation.shape[0]*Y_validation.shape[1], 3))
Y_valid_bool = np.argmax(Y_validation_temp, axis=1)

# Calculate benchmark of error
cee = CategoricalCrossentropy()
categorical_err_benchmark = cee(Y_valid_pred, Y_validation).numpy()
# print("categorical_crossentropy_benchmark: {:.2f}".format(categorical_err_benchmark))
err_dict_benchmark = classification_report(Y_valid_bool, Y_valid_pred_bool, output_dict=True)
# print(error_dict)
f1_benchmark = (float(err_dict_benchmark['0']['f1-score']), float(err_dict_benchmark['1']['f1-score']), float(err_dict_benchmark['2']['f1-score']))
print("f1_score_benchmark: (free driving-{:.2f})  (left lane change-{:.2f})  (right lane change-{:.2f})".format(f1_benchmark[0], f1_benchmark[1], f1_benchmark[2])+'\n')

# Reshape to dims (X_validation.shape[0]*X_validation.shape[1], X_validation.shape[2]) to make shuffle easier
X_reversed = X_validation.reshape(X_validation.shape[0]*X_validation.shape[1], X_validation.shape[2])

name_list = ['speed', 'acc_x', 'acc_y', 'steering_ang']
err_list = []
f1_list = []
for i in range(X_reversed.shape[1]):

    X_reversed_temp = X_reversed
    # Shuffle the selected feature column
    feature_column = X_reversed_temp[:, i]
    # Shuffle index
    p = np.random.permutation(X_reversed.shape[0])
    # Inverted shuffle index
    s = np.empty(p.size, dtype=np.int32)
    for j in np.arange(p.size):
        s[p[j]] = j

    X_reversed_temp[:, i] = feature_column[p]
    # Reshape to same dim as input for NN model
    X_permuted = X_reversed_temp.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2])
    Y_permuted = trained_model.predict(X_permuted, verbose=2)

    # Restore the X
    X_reversed_temp[:, i] = feature_column[s]
    # Calculate relative categorical crossentropy
    categorical_err = cee(Y_permuted, Y_validation).numpy()
    err_list.append(categorical_err/categorical_err_benchmark)

    # Reshape to calculate f1 score
    Y_permuted = Y_permuted.reshape((Y_permuted.shape[0] * Y_permuted.shape[1], 3))
    Y_permuted_bool = np.argmax(Y_permuted, axis=1)
    error_dict = classification_report(Y_valid_bool, Y_permuted_bool, output_dict=True, zero_division=0)
    f1 = (float(error_dict['0']['f1-score']), float(error_dict['1']['f1-score']), float(error_dict['2']['f1-score']))
    f1_list.append(f1)

# print("Categorical_crossentropy benchmark: {},  f1_score benchmark: {}".format(f1_benchmark, f1_benchmark))

for idx, item in enumerate(name_list):
    print("\"" + item + ":\"")
    print("Importance score: {:.2f}".format(err_list[idx]))
    f1_benchmark = f1_list[idx]
    print("f1_score: (free driving-{:.2f})  (left lane change-{:.2f})  (right lane change-{:.2f})".format(
        f1_benchmark[0], f1_benchmark[1], f1_benchmark[2])+'\n')