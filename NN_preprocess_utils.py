import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler


def construct_label(tem_y, new_length, label_length, class_num):
    new_y = np.zeros((new_length, class_num))
    len_origin = len(tem_y)
    # Label 1s after lane change as lane change
    label_length = round(label_length * new_length / len_origin)
    for index, label in enumerate(tem_y):
        if label != 0:
            new_index = round(index * new_length / len_origin)
            if new_index > new_length - 2:
                break
            if new_index + label_length > new_length - 1:
                if label == 1:
                    new_y[new_index:, 1] = label
                else:
                    new_y[new_index:, 2] = label
            else:
                if label == 1:
                    new_y[new_index:new_index + label_length, 1] = label
                else:
                    new_y[new_index:new_index + label_length, 2] = label
    return new_y


def construct_feature(sub_path, columns_name, window_size, new_y_length, label_length, min_size_scenarios, class_num):
    df = pd.read_csv(sub_path)

    y_ = np.zeros((df.shape[0], 1))

    # Assign the lane change left and right to different num
    for i in range(df.shape[0]):
        if int(df['lane_change_left'][i]) == 1:
            if float(df['speed'][i]) > 80:
                y_[i] = 1
            continue
        if int(df['lane_change_right'][i]) == 2:
            if float(df['speed'][i]) > 80:
                y_[i] = 2

    x = []
    y = []

    for idx, item in enumerate(y_):
        if item != 0:
            r = random.randint(0, window_size - min_size_scenarios + 1)
            temp_x = df[columns_name][idx + r - window_size:idx + r]
            temp_y = y_[idx + r - window_size:idx + r]
            temp_y = construct_label(temp_y, new_y_length, label_length, class_num)
            x.append(temp_x)
            y.append(temp_y)
            # Do this twice to synthetic more data
            r = random.randint(0, window_size - min_size_scenarios + 1)
            temp_x = df[columns_name][idx + r - window_size:idx + r]
            temp_y = y_[idx + r - window_size:idx + r]
            temp_y = construct_label(temp_y, new_y_length, label_length, class_num)
            x.append(temp_x)
            y.append(temp_y)

    examples = np.concatenate([i for i in x], axis=0)
    labels = np.concatenate([i for i in y], axis=0)

    return examples, labels


def normalization(data):
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler.transform(data), scaler.mean_, scaler.scale_
