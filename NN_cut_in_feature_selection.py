import tensorflow as tf
import numpy as np
import os
from sklearn.metrics import classification_report
from keras.models import load_model
from keras.losses import CategoricalCrossentropy

# Deactivate the GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Clear back sessions
tf.keras.backend.clear_session()

# Some hyper-parameters
# validation_split = 0.1
object_slots_num = 6

X = np.load(r'..\preprocessed_data\X_without_steering_ang_with_speed_ang_cut_in.npy')
Y = np.load(r'..\preprocessed_data\Y_without_steering_ang_with_speed_ang_cut_in.npy')


X_validation = X[:, :, :]
Y_validation = Y[:, :, :]

# Feature selection
trained_model = load_model(r'NN_data\model_19_without_steering_with_speed_GRU_cut_in_bi.h5')
trained_model.summary()
# Reshape predicted Y to dim (timesteps*m, feature_dims)
Y_valid_pred = trained_model.predict(X_validation, verbose=2)
Y_valid_pred_temp = Y_valid_pred.reshape((Y_valid_pred.shape[0] * Y_valid_pred.shape[1], 3))
Y_valid_pred_bool = np.argmax(Y_valid_pred_temp, axis=1)

# Reshape real Y to dim
Y_validation_temp = Y_validation.reshape((Y_validation.shape[0] * Y_validation.shape[1], 3))
Y_valid_bool = np.argmax(Y_validation_temp, axis=1)

# Calculate benchmark of error
cee = CategoricalCrossentropy()
categorical_err_benchmark = cee(Y_valid_pred, Y_validation).numpy()
# print("categorical_crossentropy_benchmark: {:.2f}".format(categorical_err_benchmark))
err_dict_benchmark = classification_report(Y_valid_bool, Y_valid_pred_bool, output_dict=True)
# print(error_dict)
# print(err_dict_benchmark)
f1_benchmark = (float(err_dict_benchmark['0']['f1-score']), float(err_dict_benchmark['1']['f1-score']),
                float(err_dict_benchmark['2']['f1-score']), float(err_dict_benchmark['weighted avg']['f1-score']))
print(
    "f1_score_benchmark: (free driving-{:.3f})  (left cut in-{:.3f})  (right cut in-{:.3f}) (weighted avg-{:.3f})".format(
        f1_benchmark[0], f1_benchmark[1], f1_benchmark[2], f1_benchmark[3]) + '\n')

# Reshape to dims (X_validation.shape[0]*X_validation.shape[1], X_validation.shape[2]) to make shuffle easier
X_reversed = X_validation.reshape(X_validation.shape[0] * X_validation.shape[1], X_validation.shape[2])

# name_list = ['speed', 'acc_x', 'acc_y', 'obj1_pos_x', 'obj1_pos_y', 'obj2_pos_x', 'obj2_pos_y', 'obj3_pos_x',
#              'obj3_pos_y', 'obj4_pos_x', 'obj4_pos_y', 'obj5_pos_x', 'obj5_pos_y', 'obj6_pos_x', 'obj6_pos_y']
name_list = ['speed', 'acc_x', 'acc_y', 'obj1_pos_x', 'obj1_pos_y', 'obj1_speed_x', 'obj1_speed_y', 'obj2_pos_x', 'obj2_pos_y', 'obj2_speed_x', 'obj2_speed_y', 'obj3_pos_x',
             'obj3_pos_y', 'obj3_speed_x', 'obj3_speed_y', 'obj4_pos_x', 'obj4_pos_y', 'obj4_speed_x', 'obj4_speed_y', 'obj5_pos_x', 'obj5_pos_y', 'obj5_speed_x', 'obj5_speed_y', 'obj6_pos_x', 'obj6_pos_y', 'obj6_speed_x', 'obj6_speed_y']
# name_list = ['speed', 'acc_x', 'acc_y']
# name_list = ['speed', 'acc_x', 'acc_y', 'steering_ang']
# name_list = ['speed', 'acc_x', 'acc_y', 'steering_ang', 'obj1_pos_x', 'obj1_pos_y', 'obj1_speed_x', 'obj1_speed_y', 'obj2_pos_x', 'obj2_pos_y', 'obj2_speed_x', 'obj2_speed_y', 'obj3_pos_x',
#              'obj3_pos_y', 'obj3_speed_x', 'obj3_speed_y', 'obj4_pos_x', 'obj4_pos_y', 'obj4_speed_x', 'obj4_speed_y', 'obj5_pos_x', 'obj5_pos_y', 'obj5_speed_x', 'obj5_speed_y', 'obj6_pos_x', 'obj6_pos_y', 'obj6_speed_x', 'obj6_speed_y']


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

    # Calculate relative using categorical crossentropy
    categorical_err = cee(Y_permuted, Y_validation).numpy()
    err_list.append(categorical_err / categorical_err_benchmark)

    # Reshape to calculate f1 score
    Y_permuted = Y_permuted.reshape((Y_permuted.shape[0] * Y_permuted.shape[1], 3))
    Y_permuted_bool = np.argmax(Y_permuted, axis=1)
    error_dict = classification_report(Y_valid_bool, Y_permuted_bool, output_dict=True, zero_division=0)
    f1 = (float(error_dict['0']['f1-score']), float(error_dict['1']['f1-score']), float(error_dict['2']['f1-score']),
          float(error_dict['weighted avg']['f1-score']))
    f1_list.append(f1)

    # Calculate relative importance using f1 score
    # categorical_err = cee(Y_permuted, Y_validation).numpy()
    # err_list.append(f1_benchmark[3] / float(error_dict['weighted avg']['f1-score']))

# print("Categorical_crossentropy benchmark: {},  f1_score benchmark: {}".format(f1_benchmark, f1_benchmark))

for idx, item in enumerate(name_list):
    print("\"" + item + ":\"")
    print("Importance score: {:.3f}".format(err_list[idx]))
    f1_benchmark = f1_list[idx]
    print("f1_score: (free driving-{:.3f})  (left cut in-{:.3f})  (right cut in-{:.3f}) (weighted avg-{:.3f})".format(
        f1_benchmark[0], f1_benchmark[1], f1_benchmark[2], f1_benchmark[3]) + '\n')
