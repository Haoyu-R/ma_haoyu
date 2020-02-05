import os
import pandas as pd
from NN_preprocess_utils import *

if __name__ == "__main__":

    path = r'..\preprocessed_data\test_with_steering_angle'

    #########################################Construct dataset with surrounding vehicls###################################

    # Walk through every ego data
    file_ego_list = []
    file_dynamic_list = []
    file_static_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            # Delete sub directory
            del dirs[:]
            if file.endswith("ego.csv"):
                file_ego_list.append(os.path.join(root, file))
            elif file.endswith("dynamic.csv"):
                file_dynamic_list.append(os.path.join(root, file))
            elif file.endswith("static.csv"):
                file_static_list.append(os.path.join(root, file))

    # Depend on number of features and length of each example
    window_size = 500
    # Minimal frame for one lane change scenario
    min_size_scenarios = 75
    # Number of frames to be as lane change label after one lane change
    label_length = 25
    # Which features do you want
    columns_name = ['speed', 'acc_x', 'acc_y', 'steering_ang', 'ego_line_left_distance_y', 'ego_line_right_distance_y']
    # How many classes
    class_num = 3
    # New label length: calculation based on Conv1D layer with kernel_size=7 and stride=1
    new_y_length = int((window_size - 7) / 2) + 1
    # How many object can be included in the input of NN
    object_slots_num = 6
    obj_dim = 4
    # Weather fill the hole with first or last value
    keep = True

    X_list = []
    Y_list = []

    # Loop through every files
    for idx in range(len(file_ego_list)):
        print(file_ego_list[idx])
        ego_file = pd.read_csv(file_ego_list[idx])
        dynamic_file = pd.read_csv(file_dynamic_list[idx])
        static_file = pd.read_csv(file_static_list[idx])

        x, y, exist_flag = construct_feature_lane_change_with_lane(ego_file, dynamic_file, static_file, columns_name,
                                                                   window_size,
                                                                   new_y_length, label_length, min_size_scenarios,
                                                                   class_num, object_slots_num, obj_dim, keep)

        if exist_flag:
            X_list.append(x)
            Y_list.append(y)

    X = np.concatenate([x for x in X_list], axis=0)
    X = np.nan_to_num(X)
    Y = np.concatenate([y for y in Y_list], axis=0)

    X, mean, std = normalization(X)
    # Reshape X and Y to fit the input of NN
    X = np.reshape(X, (int(X.shape[0] / window_size), window_size, len(columns_name) + obj_dim * object_slots_num))
    Y = np.reshape(Y, (int(Y.shape[0] / new_y_length), new_y_length, class_num))
    # Shuffle the X and Y
    p = np.random.permutation(X.shape[0])
    X = X[p, :, :]
    Y = Y[p, :, :]

    np.save('{}\\X_with_steering_with_lane_with_obj.npy'.format(path), X)
    np.save('{}\\Y_with_steering_with_lane_with_obj.npy'.format(path), Y)

#################################Construct dataset without surrounding vehicls#########################################

# Walk through every ego data
# file_list = []
# for root, dirs, files in os.walk(path):
#     # Remove sum directory
#     del dirs[:]
#     for file in files:
#         if file.endswith("ego.csv"):
#             file_list.append(os.path.join(root, file))
#
# # Depend on number of features and length of each example
# window_size = 500
# # Minimal frame for one lane change scenario
# min_size_scenarios = 75
# # Number of frames to be as lane change label after one lane change
# label_length = 25
# # Which features do you want
# columns_name = ['speed', 'acc_x', 'acc_y', 'steering_ang', 'ego_line_left_distance_y', 'ego_line_right_distance_y']
# # How many classes
# class_num = 3
# # New label length: calculation based on Conv1D layer with kernel_size=7 and stride=1
# new_y_length = int((window_size - 7) / 2) + 1
#
# X_list = []
# Y_list = []
#
# for sub_path in file_list:
#     print(sub_path)
#     x, y, exist_flag = construct_feature_lane_change(sub_path, columns_name, window_size, new_y_length,
#                                                      label_length, min_size_scenarios, class_num)
#     if exist_flag:
#         X_list.append(x)
#         Y_list.append(y)
#
# X = np.concatenate([x for x in X_list], axis=0)
# X = np.nan_to_num(X)
# Y = np.concatenate([y for y in Y_list], axis=0)
#
# X, mean, std = normalization(X)
# # Reshape X and Y to fit the input of NN
# X = np.reshape(X, (int(X.shape[0] / window_size), window_size, len(columns_name)))
# Y = np.reshape(Y, (int(Y.shape[0] / new_y_length), new_y_length, class_num))
# # Shuffle the X and Y
# p = np.random.permutation(X.shape[0])
# X = X[p, :, :]
# Y = Y[p, :, :]
#
# np.save('{}\\X_with_steering_with_lane.npy'.format(path), X)
# np.save('{}\\Y_with_steering_with_lane.npy'.format(path), Y)
# np.save('{}\\mean_std_without_steering_ang.npy'.format(path), mean_std)
