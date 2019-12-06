import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler


def construct_label(tem_y, new_length, label_length, class_num):
    """
    For every lane change section, construct the corresponding Y label series
    :param tem_y:
    :param new_length:
    :param label_length:
    :param class_num:
    :return:
    """
    new_y = np.zeros((new_length, class_num))
    # Use one hot categories, therefore, set the first dim to one to present free driving, later if a lane change
    # happens reset to 0 in the same section
    new_y[0, :] = 1

    len_origin = len(tem_y)
    # Label 1s after cross the change as lane change
    label_length = round(label_length * new_length / len_origin)
    for index, label in enumerate(tem_y):
        if label != 0:
            new_index = round(index * new_length / len_origin)
            if new_index > new_length - 2:
                break
            if new_index + label_length > new_length - 1:
                if label == 1:
                    # Set the dim1 to 1 to present lane change, and reset dim1 to 0
                    new_y[new_index:, 1] = 1
                    new_y[new_index:, 0] = 0
                else:
                    new_y[new_index:, 2] = 1
                    new_y[new_index:, 0] = 0
            else:
                if label == 1:
                    new_y[new_index:new_index + label_length, 1] = 1
                    new_y[new_index:new_index + label_length, 0] = 0
                else:
                    new_y[new_index:new_index + label_length, 2] = 1
                    new_y[new_index:new_index + label_length, 0] = 0
    return new_y


def construct_feature(sub_path, columns_name, window_size, new_y_length, label_length, min_size_scenarios, class_num):
    """
    Construct training examples from one ego file
    :param sub_path: The path of one ego file
    :param columns_name: The name of features in csv column
    :param window_size: length of one example
    :param new_y_length: Calculated y length based on window_size and Conv1d filter
    :param label_length: For each lane change, the length of
    :param min_size_scenarios:
    :param class_num: How many number of classes. lane change: 3 classes (free driving, turn right and turn left)
    :return: The extracted X and Y of one ego file
    """
    df = pd.read_csv(sub_path)

    y_ = np.zeros((df.shape[0], 1))

    # Assign the lane change left and right flag
    for i in range(df.shape[0]):
        if int(df['lane_change_left'][i]) == 1:
            if float(df['speed'][i]) > 80:
                y_[i] = 1
            continue
        if int(df['lane_change_right'][i]) == 2:
            if float(df['speed'][i]) > 80:
                y_[i] = 2

    # Two list to append X and Y from every example
    x = []
    y = []

    for idx, item in enumerate(y_):
        if item != 0:
            # Random a number to cut the lane change section
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

    # Concatenate all the examples from lists
    examples = np.concatenate([i for i in x], axis=0)
    labels = np.concatenate([i for i in y], axis=0)

    return examples, labels


def normalization(data):
    """
    Use sklearn to normalize the data
    :param data: Data with shape (time_steps, feature_dims)
    :return: Normalized data with it's per feature mean and standard deviation
    """
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler.transform(data), scaler.mean_, scaler.scale_
