import numpy as np
import pandas as pd
import random

def construct_feature(sub_path, feature_num, example_length, label_length):
    df = pd.read_csv(sub_path)
    a = df.index[df['lane_change_left'] == 1].tolist()
    b = df.index[df['lane_change_right'] == 1].tolist()
    return a, b


def normalization(X):
    return True


def fit_dims(X, Y):
    return True


def check_lane_change(Y_, idx):

    return counter


def construct_label(tem_y, new_y_length):
    new_y = np.zeros((1, new_y_length))
    len_origin = len(tem_y)
    label_length = round(25 * new_y_length / len_origin)
    for index, label in enumerate(tem_y):
        if label != 0:
            new_index = round(index * new_y_length / len_origin)
            if new_index == new_y_length - 2:
                break
            if new_index + label_length > new_y_length:
                new_y[new_index:] = label
            else:
                new_y[new_index:new_index+label_length] = label
    return new_y


if __name__ == "__main__":
    df = pd.read_csv(r'C:\Users\A6OJTFD\Desktop\MDM data process\processed_data\csv\0_cv_ego.csv')

    Y_ = np.zeros((df.shape[0], 1))

    # Assign the lane change left and right to different num
    for i in range(df.shape[0]):
        if df['lane_change_left'][i] == 1:
            if df['speed'][i] > 80:
                Y_[i] = 1
            continue
        if df['lane_change_right'][i] == 1:
            if df['speed'][i] > 80:
                Y_[i] = 2

    window_size = 500
    min_size_scenarios = 75
    # Depends which signals you want to have
    feature_dims = 3
    new_y_length = int((window_size-7))/2 + 1

    x = np.zeros((window_size, feature_dims))
    y = np.zeros((1, new_y_length))

    for idx, item in enumerate(Y_):
        if item != 0:
            r = random.randint(0, window_size-min_size_scenarios+1)
            temp_x = df[['speed', 'acc_x', 'acc_y']][idx+r-window_size:idx+r]
            temp_y = Y_[idx+r-window_size:idx+r]
            temp_y = construct_label(temp_y, new_y_length)
            x = np.concatenate((x, temp_x), axis=0)
            y = np.concatenate((y, temp_y), axis=0)
            # Do this twice to synthetic more data
            r = random.randint(0, window_size - min_size_scenarios + 1)
            temp_x = df[['speed', 'acc_x', 'acc_y']][idx + r - window_size:idx + r]
            temp_y = Y_[idx + r - window_size:idx + r]
            temp_y = construct_label(temp_y, new_y_length)
            x = np.concatenate((x, temp_x), axis=0)
            y = np.concatenate((y, temp_y), axis=0)






    # label = a + b
    # label.sort()
    # print(label)
    # count = 0
    # for idx, item in Y_:
    #     if item > 0:
    #         if []
