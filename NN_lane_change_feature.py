import os
import pandas as pd
from NN_preprocess_utils import *

if __name__ == "__main__":

    path = r'C:\Users\arhyr\Desktop\audi\ma_haoyu\processed_data\csv\test'
    # Walk through every ego data
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith("ego.csv"):
                file_list.append(os.path.join(root, file))

    # Depend on number of features and length of each example
    window_size = 500
    # Minimal frame for one lane change scenario
    min_size_scenarios = 75
    # Number of frames to be as lane change label after one lane change
    label_length = 25
    # Which features do you want
    columns_name = ['speed', 'acc_x', 'acc_y', 'steering_ang']
    # How many classes
    class_num = 3
    # columns_name = ['speed', 'acc_x', 'acc_y']
    # New label length: calculation based on Conv1D layer with kernel_size=7 and stride=1
    new_y_length = int((window_size-7)/2) + 1

    X_list = []
    Y_list = []

    for sub_path in file_list:
        x, y = construct_feature(sub_path, columns_name, window_size, new_y_length, label_length, min_size_scenarios, class_num)
        X_list.append(x)
        Y_list.append(y)

    X = np.concatenate([x for x in X_list], axis=0)
    Y = np.concatenate([y for y in Y_list], axis=0)

    X, mean, std = normalization(X)
    # Reshape X and Y to fit the input of NN
    X = np.reshape(X, (int(X.shape[0]/window_size), window_size, len(columns_name)))
    Y = np.reshape(Y, (int(Y.shape[0]/new_y_length), new_y_length, class_num))
    # Shuffle the X and Y
    p = np.random.permutation(X.shape[0])
    X = X[p, :, :]
    Y = Y[p, :, :]

    mean_std = np.reshape(np.concatenate((mean, std), axis=0), (2, len(columns_name)))

    np.save('{}\\X.npy'.format(path), X)
    np.save('{}\\Y.npy'.format(path), Y)
    np.save('{}\\mean_std.npy'.format(path), mean_std)


