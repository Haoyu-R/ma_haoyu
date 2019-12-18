import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from visualization_utils import *


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
    new_y[:, 0] = 1

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


def construct_feature_lane_change(sub_path, columns_name, window_size, new_y_length, label_length, min_size_scenarios, class_num):
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
        if int(df['lane_change_right'][i]) == 1:
            if float(df['speed'][i]) > 80:
                y_[i] = 2

    # Two list to append X and Y from every example
    x = []
    y = []

    for idx, item in enumerate(y_):
        if item != 0:
            # Random a number to cut the lane change section
            r = random.randint(0, window_size - min_size_scenarios + 1)
            # Prevent exceed range
            if r+idx >= df.shape[0]-1 or idx + r - window_size < 0:
                continue
            temp_x = df[columns_name][idx + r - window_size:idx + r]
            temp_y = y_[idx + r - window_size:idx + r]
            if len(temp_y) == 0:
                print(idx)
                print(len(temp_x))
                continue
            temp_y = construct_label(temp_y, new_y_length, label_length, class_num)
            x.append(temp_x)
            y.append(temp_y)

            # Do this twice to synthetic more data
            r = random.randint(0, window_size - min_size_scenarios + 1)
            if r+idx >= df.shape[0]-1 or idx + r - window_size < 0:
                continue
            temp_x = df[columns_name][idx + r - window_size:idx + r]
            temp_y = y_[idx + r - window_size:idx + r]
            if len(temp_y) == 0:
                print(idx)
                print(len(temp_x))
                continue
            temp_y = construct_label(temp_y, new_y_length, label_length, class_num)
            x.append(temp_x)
            y.append(temp_y)

    # Concatenate all the examples from lists
    if len(x) == 0:
        return -1, -1, False
    else:
        examples = np.concatenate([i for i in x], axis=0)
        labels = np.concatenate([i for i in y], axis=0)

        return examples, labels, True


def normalization(data):
    """
    Use sklearn to normalize the data
    :param data: Data with shape (time_steps, feature_dims)
    :return: Normalized data with it's per feature mean and standard deviation
    """
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler.transform(data), scaler.mean_, scaler.scale_


def normalization_cut_in(data, columns_name, object_slots_num):
    """
    Use sklearn to normalize the data
    :param data: Data with shape (time_steps, feature_dims)
    :return: Normalized data with it's per feature mean and standard deviation
    """
    ego_column_num = len(columns_name)
    data[:, :ego_column_num], _, _ = normalization(data[:, :ego_column_num])
    other_column = data[:, ego_column_num:]

    # Append objects from all channels to two columns and calculate its mean and scale
    new_other_column = np.empty((0, 2), float)
    for i in range(object_slots_num):
        temp = other_column[:, i * 2:i * 2 + 2]
        new_other_column = np.append(new_other_column, temp, axis=0)

    scaler = StandardScaler()
    scaler.fit(new_other_column)
    # For every object channel standardization
    for i in range(object_slots_num):
        data[:, ego_column_num + 2*i:ego_column_num + 2*i + 2] = (data[:, ego_column_num + 2*i:ego_column_num + 2*i + 2] - scaler.mean_)/scaler.scale_

    return data


def cut_in_list(dynamic_df, ego_df):
    """
    Find out where left and right cut in happen. 1 = left cut in; 2 = right cut in
    :param dynamic_df:
    :param ego_df:
    :return:
    """
    y_ = np.zeros((ego_df.shape[0], 1))
    for i in range(dynamic_df.shape[0]):
        # At the "cut in" moment, "cut in" object should not be too far from ego vehicle
        if (dynamic_df['cut_in_left'][i] == 1) and (dynamic_df['pos_x'][i] < 60):
            frame_num = dynamic_df['frame'][i]
            if ego_df['speed'][int(frame_num)] > 80:
                y_[frame_num] = 1
            continue
        if (dynamic_df['cut_in_right'][i] == 1) and (dynamic_df['pos_x'][i] < 60):
            frame_num = dynamic_df['frame'][i]
            if ego_df['speed'][int(frame_num)] > 80:
                y_[frame_num] = 2
    return y_


def construct_cut_in_X(start_frame, end_frame, ego_file, dynamic, static, object_slots_num, window_size, columns_name):
    """
    For every cut in scenarios window construct corresponding input X window
    :param start_frame:
    :param end_frame:
    :param ego_file:
    :param dynamic:
    :param static:
    :param object_slots_num:
    :param window_size:
    :param columns_name:
    :return:
    """
    # The first part of temp_x is the ego status, the second part is slots of surrounding objects, each slot include
    # two dimension: x and y
    temp_x = np.zeros((window_size, len(columns_name)+object_slots_num*2))
    # Get ego vehicle info
    temp_x[:, :len(columns_name)] = ego_file[columns_name][start_frame:end_frame]
    # check which objects are in the current time window
    i = 0
    for obj in dynamic:
        # Choose only first "object_slots_num" showed object
        if i >= object_slots_num:
            break
        obj_id = obj['obj_id']
        obj_static = static[obj_id]
        initial_obj_frame = obj_static['initial_frame']
        total_obj_frames = obj_static['total_frames']
        # Actually here is end_obj_frame + 1
        end_obj_frame = initial_obj_frame + total_obj_frames
        # The involved surrounding vehicle should also be not far from ego vehicle
        if initial_obj_frame < end_frame and start_frame < end_obj_frame:
            # Four situations of each object frame in the window related to start_frame and end_frame of window
            if initial_obj_frame <= start_frame and end_obj_frame < end_frame and obj['pos_x'][start_frame - initial_obj_frame] < 80 and abs(obj['pos_y'][start_frame - initial_obj_frame]) < 15:
                temp_x[:end_obj_frame - start_frame, len(columns_name) + i * 2] = obj['pos_x'][start_frame - initial_obj_frame:]
                temp_x[:end_obj_frame - start_frame, len(columns_name) + 1 + i * 2] = obj['pos_y'][start_frame - initial_obj_frame:]
                temp_x[end_obj_frame - start_frame:, len(columns_name) + i * 2] = obj['pos_x'][-1]
                temp_x[end_obj_frame - start_frame:, len(columns_name) + 1 + i * 2] = obj['pos_y'][-1]
                i += 1
                continue
            if initial_obj_frame > start_frame and end_obj_frame < end_frame and obj['pos_x'][0] < 80 and abs(obj['pos_y'][0]) < 15:
                temp_x[initial_obj_frame-start_frame:end_obj_frame - start_frame, len(columns_name) + i * 2] = obj['pos_x'][:]
                temp_x[initial_obj_frame-start_frame:end_obj_frame - start_frame, len(columns_name) + 1 + i * 2] = obj['pos_y'][:]
                temp_x[:initial_obj_frame - start_frame, len(columns_name) + i * 2] = obj['pos_x'][0]
                temp_x[:initial_obj_frame - start_frame, len(columns_name) + 1 + i * 2] = obj['pos_y'][0]
                temp_x[end_obj_frame - start_frame:, len(columns_name) + i * 2] = obj['pos_x'][-1]
                temp_x[end_obj_frame - start_frame:, len(columns_name) + 1 + i * 2] = obj['pos_y'][-1]
                i += 1
                continue
            if initial_obj_frame > start_frame and end_obj_frame >= end_frame and obj['pos_x'][0] < 80 and abs(obj['pos_y'][0]) < 15:
                temp_x[initial_obj_frame-start_frame:, len(columns_name) + i * 2] = obj['pos_x'][:end_frame - initial_obj_frame]
                temp_x[initial_obj_frame-start_frame:, len(columns_name) + 1 + i * 2] = obj['pos_y'][:end_frame - initial_obj_frame]
                temp_x[:initial_obj_frame-start_frame, len(columns_name) + i * 2] = obj['pos_x'][0]
                temp_x[:initial_obj_frame-start_frame, len(columns_name) + 1 + i * 2] = obj['pos_y'][0]
                i += 1
                continue
            if initial_obj_frame < start_frame and end_obj_frame > end_frame and obj['pos_x'][start_frame - initial_obj_frame] < 80 and abs(obj['pos_y'][start_frame - initial_obj_frame]) < 15:
                temp_x[:, len(columns_name) + i * 2] = obj['pos_x'][start_frame - initial_obj_frame:end_frame-initial_obj_frame]
                temp_x[:, len(columns_name) + 1 + i * 2] = obj['pos_y'][start_frame - initial_obj_frame:end_frame-initial_obj_frame]
                i += 1
                continue
    return temp_x


def construct_feature_cut_in(ego_file, dynamic_file, static_file, columns_name, window_size, new_y_length, label_length,
                                         min_size_scenarios, class_num, object_slots_num):
    """
    Construct the input and reference label for NN from every file group
    :param ego_file: ego_file from process_main
    :param dynamic_file: dynamic_file from process_main
    :param static_file: static_file from process_main
    :param columns_name: names of columns of ego state that include in input
    :param window_size: How long should each input windows has
    :param new_y_length: How long should each output has
    :param label_length: How long should a positive label has
    :param min_size_scenarios: How long should a scenario at least has
    :param class_num: How many classes are there
    :param object_slots_num: At most how many objects want to have in one window
    :return: Input to NN and reference output
    """
    cut_in_ls = cut_in_list(dynamic_file, ego_file)
    index_ls = [idx for idx, item in enumerate(cut_in_ls) if item != 0]
    # Use group by to group the objects in dict
    dynamic = process_dynamic(dynamic_file)
    static = process_static(static_file)
    # Two list to append X and Y from every example
    x = []
    y = []

    if len(index_ls) == 0:
        return -1, -1, False

    for idx, item in enumerate(cut_in_ls):
        if item != 0:
            # Random a number to cut the cut in section
            r = random.randint(0, window_size - min_size_scenarios + 1)
            # Prevent exceed range
            if idx + r >= ego_file.shape[0]-1 or idx + r - window_size < 0:
                continue
            temp_x = construct_cut_in_X(idx + r - window_size, idx + r, ego_file, dynamic, static, object_slots_num, window_size, columns_name)
            # Select the part from whole cut_in_ls where cut in happen
            temp_y = cut_in_ls[idx + r - window_size:idx + r]
            temp_y = construct_label(temp_y, new_y_length, label_length, class_num)
            x.append(temp_x)
            y.append(temp_y)

            # Do this twice to synthetic more data
            r = random.randint(0, window_size - min_size_scenarios + 1)
            # Prevent exceed range
            if r + idx >= ego_file.shape[0] - 1 or idx + r - window_size < 0:
                continue
            temp_x = construct_cut_in_X(idx + r - window_size, idx + r, ego_file, dynamic, static, object_slots_num, window_size, columns_name)
            temp_y = cut_in_ls[idx + r - window_size:idx + r]
            temp_y = construct_label(temp_y, new_y_length, label_length, class_num)
            x.append(temp_x)
            y.append(temp_y)

    # Concatenate all the examples from lists
    if len(x) == 0:
        return -1, -1, False
    else:
        examples = np.concatenate([i for i in x], axis=0)
        labels = np.concatenate([i for i in y], axis=0)
        return examples, labels, True



