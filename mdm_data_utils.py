import numpy as np
import pandas as pd
import math
import os
import glob
from pathlib import Path


def csv_to_pickle(trace_path):
    """
    Read the csv data from the given trace path
    (Make sure to run with a 64 bit python if csv is big)
    :param trace_path: url of a cav data
    :return: No return, save csv as pickle file in pickle_file folder
    """
    if os.path.isfile(trace_path):
        with open(trace_path) as file:
            pd.read_csv(file).to_pickle(r"..\pickle\\" + str(Path(trace_path).resolve().stem))
    elif os.path.isdir(trace_path):
        files = glob.glob(trace_path + r'\*.csv')
        for file in files:
            with open(file) as tmp_file:
                pd.read_csv(tmp_file).to_pickle(r"..\pickle\\" + str(Path(file).resolve().stem))
    else:
        print("It's not a valid trace path")
        return 0


def heading_interpolation(df_heading):
    """
    Calculate the interpolation of heading column
    :param df_heading: heading and timestamp column
    :return: interpolated columns of input
    """
    # dataframeHeading = dataFrame[['timestamp', headingColumn[0]]]
    # dataframeHeadingNotNull = dataframeHeading.dropna(thresh=2)
    df_timestamp = df_heading['frame']
    df_heading_not_null = df_heading.dropna(thresh=2)
    tmp_np_object = np.rad2deg(np.unwrap(np.deg2rad(
        df_heading_not_null['heading'])))  # preprocessing heading of car in case 1 to 359 grade change happen
    df_to_merge = pd.DataFrame(columns=['frame', 'heading'])
    df_to_merge['frame'] = df_heading_not_null['frame']
    df_to_merge['heading'] = tmp_np_object
    df_heading = pd.merge(left=df_timestamp, right=df_to_merge, on="frame", how='left')  # merge preprocessed heading
    df_heading = df_heading.interpolate(method='linear', limit_direction='both',
                                        axis=0).ffill().bfill()  # linear interpolation for heading
    df_heading['heading'] %= 360  # Get correct values for heading
    return df_heading['heading']


def frame_count(df, check_valid_index):
    """
    Calculate the first and last valid frame in dataframe
    :param check_valid_index: if do the valid index check
    :param df: data frame
    :return: index of first and last valid frame
    """
    max_frames = df.shape[0]
    first_valid_index = -1
    last_valid_index = -1
    if not check_valid_index:
        return 0, max_frames - 1, False, ""

    speed_column = [col for col in df.columns if 'ESP_v_Signal' in col]
    line_left_column = [col for col in df.columns if 'BV1_LIN_01_EndeX' in col]
    line_right_column = [col for col in df.columns if 'BV1_LIN_02_EndeX' in col]

    if len(line_right_column) == 0 or len(line_left_column) == 0:
        return 0, max_frames - 1, True, "No line information"

    for timestamp in range(max_frames):
        if (not math.isnan(df[speed_column[0]][timestamp])) and (
                float(df[speed_column[0]][timestamp]) > 0.1) and not math.isnan(
            df[line_left_column[0]][timestamp]) and not math.isnan(df[line_right_column[0]][timestamp]):
            # set to speed > 0.1 in case some fluctuation and only valid if line info exists
            first_valid_index = timestamp
            break
    for timestamp in range(max_frames - 1, -1, -1):

        if (not math.isnan(df[speed_column[0]][timestamp])) and (
                float(df[speed_column[0]][timestamp]) > 0.1) and not math.isnan(
            df[line_left_column[0]][timestamp]) and not math.isnan(df[line_right_column[0]][timestamp]):
            last_valid_index = timestamp
            break

    if first_valid_index >= 0 and last_valid_index > 0:
        return first_valid_index, last_valid_index, False, ""

    else:
        return first_valid_index, last_valid_index, True, "No valid movement data"


def steering_symbol(a, b):
    """
    add the right symbol to steering angle
    :param a: Columns of steering angle
    :param b: Columns of symbol of steering angle
    :return:
    """
    if b > 0:
        return a
    else:
        return -a


def new_ego_df(df, first_valid_index, last_valid_index):
    """
    Generate the valid dataframe for ego info in new data structure
    :param df: row datafram
    :param first_valid_index: index of first un-zero value of speed
    :param last_valid_index: index of last un-zero value of speed
    :return: reformulated data with desired columns for ego line info and a flag to show if the data has steering ang info
    """

    # time_frame = [col for col in df.columns if 'timestamp' in col]
    speed_column = [col for col in df.columns if 'ESP_v_Signal' in col]
    heading_column = [col for col in df.columns if 'ND_Heading' in col]
    acc_x_column = [col for col in df.columns if 'SARA_Accel_X_010' in col]
    acc_y_column = [col for col in df.columns if 'SARA_Accel_Y_010' in col]
    steering_ang_column = [col for col in df.columns if 'LWI_Lenkradwinkel' in col]
    steering_ang_symbol_column = [col for col in df.columns if 'LWI_VZ_Lenkradwinkel' in col]

    # In case some data doesn't have steering angle
    if_steering_ang = False
    if len(steering_ang_column) > 0:
        if_steering_ang = True
        steering_df = df.loc[first_valid_index:last_valid_index, [steering_ang_column[0],
                                                                  steering_ang_symbol_column[0]
                                                                  ]].reset_index(drop=True)
        steering_df.columns = ['steering_ang',
                               'steering_ang_symbol']
        # Some values in steering ang are NAN, interpolate them
        steering_df['steering_ang'] = steering_df['steering_ang'].interpolate(method='linear', limit_direction='both',
                                                                              axis=0).ffill().bfill()
        steering_df['steering_ang_symbol'] = steering_df['steering_ang_symbol'].interpolate(method='linear',
                                                                                            limit_direction='both',
                                                                                            axis=0).ffill().bfill()
        steering_ang_column = list(steering_df['steering_ang'])

        # Careful, steering angle data in new e-tron data already has symbol
        symbol_flag = True
        a = np.absolute(steering_ang_column)
        print(np.max(a))
        for ang in steering_ang_column:
            if abs(ang) > 14:
                steering_ang_column = [ang * 0.0174533 for ang in steering_ang_column]
                break

        for ang in steering_ang_column:
            if ang < 0:
                symbol_flag = False
                break

        if symbol_flag:
            steering_ang_symbol_column = list(steering_df['steering_ang_symbol'])
            tem_zip = zip(steering_ang_column, steering_ang_symbol_column)
            steering_ang_column = [steering_symbol(float(a), float(b)) for a, b in tem_zip]

    ego_line_left_begin_x_column = [col for col in df.columns if 'BV1_LIN_01_BeginnX' in col]
    ego_line_left_end_x_column = [col for col in df.columns if 'BV1_LIN_01_EndeX' in col]
    ego_line_left_distance_y_column = [col for col in df.columns if 'BV1_LIN_01_AbstandY' in col]
    ego_line_left_curv_column = [col for col in df.columns if 'BV1_LIN_01_HorKruemm' in col and 'Aend' not in col]
    ego_line_right_begin_x_column = [col for col in df.columns if 'BV1_LIN_02_BeginnX' in col]
    ego_line_right_end_x_column = [col for col in df.columns if 'BV1_LIN_02_EndeX' in col]
    ego_line_right_distance_y_column = [col for col in df.columns if 'BV1_LIN_02_AbstandY' in col]
    ego_line_right_curv_column = [col for col in df.columns if 'BV1_LIN_02_HorKruemm' in col and 'Aend' not in col]

    if if_steering_ang:
        new_df = df.loc[first_valid_index:last_valid_index, [speed_column[0], heading_column[0],
                                                             acc_x_column[0], acc_y_column[0],
                                                             ego_line_left_begin_x_column[0],
                                                             ego_line_left_end_x_column[0],
                                                             ego_line_left_distance_y_column[0],
                                                             ego_line_left_curv_column[0],
                                                             ego_line_right_begin_x_column[0],
                                                             ego_line_right_end_x_column[0],
                                                             ego_line_right_distance_y_column[0],
                                                             ego_line_right_curv_column[0]]].reset_index(drop=True)
        new_df.columns = ['speed', 'heading', 'acc_x', 'acc_y',
                          'ego_line_left_begin_x', 'ego_line_left_end_x', 'ego_line_left_distance_y',
                          'ego_line_left_curv',
                          'ego_line_right_begin_x', 'ego_line_right_end_x', 'ego_line_right_distance_y',
                          'ego_line_right_curv']
        new_df['steering_ang'] = steering_ang_column
    else:
        new_df = df.loc[first_valid_index:last_valid_index, [speed_column[0], heading_column[0],
                                                             acc_x_column[0], acc_y_column[0],
                                                             ego_line_left_begin_x_column[0],
                                                             ego_line_left_end_x_column[0],
                                                             ego_line_left_distance_y_column[0],
                                                             ego_line_left_curv_column[0],
                                                             ego_line_right_begin_x_column[0],
                                                             ego_line_right_end_x_column[0],
                                                             ego_line_right_distance_y_column[0],
                                                             ego_line_right_curv_column[0]]].reset_index(drop=True)
        new_df.columns = ['speed', 'heading', 'acc_x', 'acc_y',
                          'ego_line_left_begin_x', 'ego_line_left_end_x', 'ego_line_left_distance_y',
                          'ego_line_left_curv',
                          'ego_line_right_begin_x', 'ego_line_right_end_x', 'ego_line_right_distance_y',
                          'ego_line_right_curv']

    tem_frame = range(len(new_df))  # to add a new timestamp in the front of df
    new_df.insert(loc=0, column='frame', value=tem_frame)
    tem_heading_series = heading_interpolation(
        new_df.loc[:, ['frame', 'heading']])  # to interpolate heading value, because there are too many nan value
    new_df['heading'] = tem_heading_series
    return new_df, if_steering_ang


def num_to_string(num):
    """
    Padding the slot number to fit the name in recording
    :param num: slot number
    :return: padding number in string
    """
    if num < 10:
        return '0' + str(math.floor(num))
    else:
        return str(math.floor(num))


def gaussian_func(gau_window, sigma):
    # For gaussian filter
    return 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-(gau_window ** 2) / (2 * sigma ** 2))


def most_common(lst):
    # Select the most possible class for the object
    return max(set(lst), key=lst.count)


def check_valid_vehicle(vehicle_speed_x_list, pos_y_list):
    """
    Check if the given object is in the opposite driving direction or it's even not on the highway (maybe noise or on
    other adjecent road)
    :param vehicle_speed_x_list: speed list in x direction of one object
    :param pos_y_list: y_position list of one object
    :return: return Ture if is in the opposite direction
    """
    m = len(vehicle_speed_x_list)
    speed_array = np.array(vehicle_speed_x_list)
    pos_y_array = np.array(pos_y_list)
    # Check if the vehicle is on the road
    pos_y_list_plus_15 = [pos_y_array > 14]
    pos_y_list_minus_15 = [pos_y_array < -14]
    # Check if the vehicle is in the same driving direction
    bool_array_minus_speed = [speed_array < 0]

    # In valid if too many negative values in speed list or object has too many y coordinates that far away from ego
    if m < 5 * np.sum(bool_array_minus_speed) or m < 2 * np.sum(pos_y_list_minus_15) or m < 2 * np.sum(
            pos_y_list_plus_15):
        # print(m, np.sum(bool_array_minus_speed), np.sum(pos_y_list_minus_15), np.sum(pos_y_list_plus_15))
        return True
    else:
        return False


def construct_object_dict(class_list, width_list,
                          # length_list
                          pos_x_list, pos_y_list, speed_x_list, speed_y_list,
                          index, first_valid_index, frame, count):
    """
    Calculate and adjust all the information of one object
    :param class_list: class of one object in all frames
    :param width_list: width of one object in all frames
    # :param length_list: length of one object in all frames
    :param pos_x_list: x coordinate of one object in all frames
    :param pos_y_list: y coordinate of one object in all frames
    :param speed_x_list: x speed of one object in all frames
    :param speed_y_list: y speed of one object in all frames
    :param index: First frame of object
    :param first_valid_index: First valid frame of data
    :param frame: Last frame of the object
    :param count: the id of last object
    :return: current object id, dynamic info of one object, static info of one object
    """
    class_dict = {0: 'Unknown', 1: 'Dynamic', 2: 'VRU', 3: 'Pedestrian', 4: 'Pedestrian_Group', 5: 'Cyclist',
                  6: 'Motorcycle', 7: 'Car', 8: 'At_least_4_wheels', 9: 'Truck', 10: 'Bicycle', 11: 'Animal',
                  12: 'At_least_2_wheels', 13: 'Undefined', 14: 'Undefined', 15: 'init'}
    # print(len(pos_x_list), len(pos_y_list), len(speed_y_list), len(speed_x_list), index, first_valid_index, frame)
    obj_class = most_common(class_list)

    width = max(list(map(float, width_list)))
    if math.isnan(width):
        width = 0
    # length = max(list(map(float, length_list)))
    # if math.isnan(length):
    #     length = 0

    initial_frame = index
    total_frames = frame - index
    timestamp_list = list(range(index - first_valid_index, frame - first_valid_index))

    count += 1
    id_lst = [count] * (frame - index)

    # Apply a gaussian filter on the x and y position of object to make trajectory smooth
    sigma = 2
    gau_window = np.linspace(-2.7 * sigma, 2.7 * sigma, 6 * sigma)  # Sliding window will have 6*sigma length
    gau_mask = gaussian_func(gau_window, sigma)
    # window_length = 5
    # gau_mask = np.ones(window_length)/window_length
    pos_x_list_tmp = np.convolve(pos_x_list, gau_mask, 'same')
    pos_y_list_tmp = np.convolve(pos_y_list, gau_mask, 'same')
    # To prevent edge effect, don't take the edge coordinates of object
    pos_x_list[5:len(pos_x_list) - 5] = pos_x_list_tmp[5:len(pos_x_list) - 5]
    pos_y_list[5:len(pos_y_list) - 5] = pos_y_list_tmp[5:len(pos_y_list) - 5]

    dict_dynamic = {'frame': timestamp_list, 'obj_id': id_lst, 'pos_x': pos_x_list, 'pos_y': pos_y_list,
                    'speed_x': speed_x_list, 'speed_y': speed_y_list}

    dict_static = {'obj_id': [count], 'obj_class': [class_dict[int(obj_class)]],
                   # 'length': [int(length)],
                   'width': [int(width)], 'initial_frame': [initial_frame - first_valid_index],
                   'total_frames': [total_frames]}

    return count, pd.DataFrame(dict_dynamic), pd.DataFrame(dict_static)


def get_object_info(df, new_data, index, slot, obj_id, first_string, second_string, count, first_valid_index,
                    last_valid_index,
                    minimum_frames):
    """
    trace the information of one object in all the slots
    :param new_data: if is the new e-tron data, because the column name is different
    :param df: raw data frame
    :param index: the first index that this object shows up
    :param slot: the first slot that this object shows up
    :param obj_id: current id of the object
    :param first_string: for column adaption
    :param second_string: for column adaption
    :param minimum_frames: minimum frames that one object should have, otherwise the object is invalid
    :param count: how many objects have been detected
    :param last_valid_index: First valid frame in data
    :param first_valid_index: Last valid frame in data
    :return: the dynamic and static information of the object and count += 1 if one object is detected
    """
    pos_x_list = []
    pos_y_list = []
    class_list = []
    width_list = []
    # length_list = []
    speed_x_list = []
    speed_y_list = []

    if new_data:
        id_str_end = '_ID'
        positionX_str_end = '_PositionX'
        positionY_str_end = '_PositionY'
        class_str_end = '_Klasse'
        width_str_end = '_Breite'
        speedX_str_end = '_GeschwX'
        speedY_str_end = '_GeschwY'
        his_str_end = '_Historie'
    else:
        id_str_end = '_ID)'
        positionX_str_end = '_PositionX)'
        positionY_str_end = '_PositionY)'
        class_str_end = '_Klasse)'
        width_str_end = '_Breite)'
        speedX_str_end = '_GeschwX)'
        speedY_str_end = '_GeschwY)'
        his_str_end = '_Historie)'

    for frame in range(index, df.shape[0]):
        num_slot = num_to_string(slot)
        if df[first_string + str(num_slot) + second_string + str(num_slot) + id_str_end][frame] == obj_id:
            tmp_pos_x = df[first_string + str(num_slot) + second_string + str(num_slot) + positionX_str_end][frame]
            pos_x_list.append(tmp_pos_x)
            tmp_pos_y = df[first_string + str(num_slot) + second_string + str(num_slot) + positionY_str_end][frame]
            pos_y_list.append(tmp_pos_y)
            tmp_class = df[first_string + str(num_slot) + second_string + str(num_slot) + class_str_end][frame]
            class_list.append(tmp_class)
            tmp_width = df[first_string + str(num_slot) + second_string + str(num_slot) + width_str_end][frame]
            width_list.append(tmp_width)
            # tmp_length = df[first_string + str(num_slot) + second_string + str(num_slot) + '_Laenge)'][frame]
            # length_list.append(tmp_length)
            tmp_speed_x = df[first_string + str(num_slot) + second_string + str(num_slot) + speedX_str_end][frame]
            speed_x_list.append(tmp_speed_x * 3.6)
            tmp_speed_y = df[first_string + str(num_slot) + second_string + str(num_slot) + speedY_str_end][frame]
            speed_y_list.append(tmp_speed_y * 3.6)
            if frame == last_valid_index:
                if frame - index <= minimum_frames:  # only take objects which show up longer than 20 frames
                    return False, False, False
                elif check_valid_vehicle(speed_x_list, pos_y_list):
                    return False, False, False
                else:
                    return construct_object_dict(class_list, width_list,
                                                 # length_list,
                                                 pos_x_list, pos_y_list, speed_x_list,
                                                 speed_y_list, index, first_valid_index, frame + 1, count)
        else:
            flag = 0  # monitor if the trajectory of object is found in other slot
            for new_slot in range(1, 11):
                num_new_slot = num_to_string(new_slot)
                if df[first_string + str(num_new_slot) + second_string + str(num_new_slot) + id_str_end][
                    frame] == obj_id and \
                        df[first_string + str(num_new_slot) + second_string + str(num_new_slot) + his_str_end][
                            frame] != 0:
                    slot = new_slot
                    tmp_pos_x = \
                        df[first_string + str(num_new_slot) + second_string + str(num_new_slot) + positionX_str_end][
                            frame]
                    pos_x_list.append(tmp_pos_x)
                    tmp_pos_y = \
                        df[first_string + str(num_new_slot) + second_string + str(num_new_slot) + positionY_str_end][
                            frame]
                    pos_y_list.append(tmp_pos_y)
                    tmp_class = \
                        df[first_string + str(num_new_slot) + second_string + str(num_new_slot) + class_str_end][
                            frame]
                    class_list.append(tmp_class)
                    tmp_width = \
                        df[first_string + str(num_new_slot) + second_string + str(num_new_slot) + width_str_end][
                            frame]
                    width_list.append(tmp_width)
                    # tmp_length = df[first_string + str(num_new_slot) + second_string + str(num_new_slot) + '_Laenge)'][
                    #     frame]
                    # length_list.append(tmp_length)
                    tmp_speed_x = \
                        df[first_string + str(num_new_slot) + second_string + str(num_new_slot) + speedX_str_end][frame]
                    speed_x_list.append(tmp_speed_x * 3.6)
                    tmp_speed_y = \
                        df[first_string + str(num_new_slot) + second_string + str(num_new_slot) + speedY_str_end][frame]
                    speed_y_list.append(tmp_speed_y * 3.6)
                    flag = 1
                    break
            if flag == 0:
                if frame - index <= minimum_frames:  # only take objects which show up longer than 30 frames
                    return False, False, False
                elif check_valid_vehicle(speed_x_list, pos_y_list):
                    return False, False, False
                else:
                    return construct_object_dict(class_list, width_list,
                                                 # length_list,
                                                 pos_x_list, pos_y_list,
                                                 speed_x_list, speed_y_list, index, first_valid_index, frame, count)


def new_objects_df(df, first_valid_index, last_valid_index, file_number, minimum_frames=20):
    """
    Calculate the dynamic and static info of surrounding objects into 2 df
    :param file_number: the index of current file
    :param df: raw df
    :param first_valid_index: the index of first valid frame in raw df
    :param last_valid_index: the index of last valid frame in raw df
    :param minimum_frames: the minimum frame number that a valid object should have
    :return: the dynamic and static df of all objects
    """

    # Check if this is the new E-Tron data, because the column name of new data is different
    new_data = False

    if 'avg(FR1B::SDF2_Objekt_01::SDF2_Obj_01_ID)' in df.columns:
        first_string_sdf = 'avg(FR1B::SDF2_Objekt_'
        first_string_cv = 'avg(FR1A::BV2_Objekt_'
        second_string_sdf = '::SDF2_Obj_'
        second_string_cv = '::BV2_Obj_'
        history_str_end = '_Historie)'
        id_str_end = '_ID)'
    else:
        new_data = True
        first_string_sdf = 'FR1B::SDF2_Objekt_'
        first_string_cv = 'FR1A::BV2_Objekt_'
        second_string_sdf = '::SDF2_Obj_'
        second_string_cv = '::BV2_Obj_'
        history_str_end = '_Historie'
        id_str_end = '_ID'

    is_SDF = False

    # Check the object data is based only on computer vision or sensor data fusion
    counter = 0
    if new_data:
        for i in range(len(df['FR1B::SDF2_Objekt_01::SDF2_Obj_01_ID'])):
            if not math.isnan(df['FR1B::SDF2_Objekt_01::SDF2_Obj_01_ID'][i]):
                counter += 1
    else:
        for i in range(len(df['avg(FR1B::SDF2_Objekt_01::SDF2_Obj_01_ID)'])):
            if not math.isnan(df['avg(FR1B::SDF2_Objekt_01::SDF2_Obj_01_ID)'][i]):
                counter += 1

    # Pick SDF data
    if counter > 0:
        first_string = first_string_sdf
        second_string = second_string_sdf
        is_SDF = True
    # Pick CV data
    else:
        first_string = first_string_cv
        second_string = second_string_cv

    count = 0
    all_objects_dynamic = pd.DataFrame()
    all_objects_static = pd.DataFrame()
    for index in range(first_valid_index, last_valid_index - minimum_frames):
        # default require that one object should have minimum 20 time frames
        # Display the progress rate in console
        if index % 200 == 0:
            print('Prepossessing the {}.data: {:.2f} %'.format(str(file_number + 1), (
                    (index - first_valid_index) / (last_valid_index - first_valid_index)) * 100))
        # Loop through every slots
        for slot in range(1, 11):
            num_slot = num_to_string(slot)
            history = df[first_string + str(num_slot) + second_string + str(num_slot) + history_str_end][index]
            obj_id = df[first_string + str(num_slot) + second_string + str(num_slot) + id_str_end][index]
            if not math.isnan(history) and history == 0 and not math.isnan(obj_id):
                tem_count, object_dynamic_info, object_static_info = get_object_info(df, new_data, index, slot, obj_id,
                                                                                     first_string, second_string, count,
                                                                                     first_valid_index,
                                                                                     last_valid_index, minimum_frames)
                if not tem_count:
                    continue
                else:
                    count = tem_count
                    all_objects_dynamic = all_objects_dynamic.append(object_dynamic_info, ignore_index=True)
                    all_objects_static = all_objects_static.append(object_static_info, ignore_index=True)
    return all_objects_static, all_objects_dynamic, is_SDF


def lane_change_conflict(arr):
    """
    construct a line change state array which to control cut in will not be labelled later during lane change
    :param arr:
    :return:
    """
    new_arr = np.zeros((len(arr)))
    for idx, item in enumerate(arr):
        if item > 0:
            first_idx = 0 if idx - 60 < 0 else idx - 60
            last_idx = len(arr) if idx + 60 > len(arr) else idx + 60
            new_arr[first_idx:idx] = 1
            new_arr[idx:last_idx] = 1
    return new_arr


def check_valid_line(ego_frames, num):
    if math.isnan(ego_frames['ego_line_left_begin_x'][num]) or math.isnan(
            ego_frames['ego_line_left_end_x'][num]) or math.isnan(
        ego_frames['ego_line_left_distance_y'][num]) or math.isnan(
        ego_frames['ego_line_left_curv'][num]) or math.isnan(
        ego_frames['ego_line_right_begin_x'][num]) or math.isnan(
        ego_frames['ego_line_right_end_x'][num]) or math.isnan(
        ego_frames['ego_line_right_distance_y'][num]) or math.isnan(ego_frames['ego_line_right_curv'][num]):
        return False
    else:
        return True


def cut_in_detection(ego_frames, object_frames, lane_change_array):
    """
    Label the cut in
    :param ego_frames: a segment of ego time frame when the object also exists
    :param object_frames: a selected object
    :param lane_change_array: labeled time frame of lane change
    :return: left_cut_count: how many left cut in
             right_cut_count: how many right cut in
             cut_in_right_dynamic: labeled time frame of right cut in
             cut_in_left_dynamic: Labeled time frame of left cut in
    """
    # Two label array to be return
    cut_in_left_dynamic = np.zeros(ego_frames.shape[0])
    cut_in_right_dynamic = np.zeros(ego_frames.shape[0])
    # List to record the time frame when cut in finished
    cut_in_finish_left = []
    cut_in_finish_right = []
    # Count how many times object carry out cut in
    left_cut_count = 0
    right_cut_count = 0
    # Flag to check if the last frame of the object is out of ego line area
    left_out_flag = False
    right_out_flag = False
    # Counter to check if the last few frames of the object is out of ego line area, 0 - inside/unknown, 1 - outside
    out_limit = 40
    left_out_list = np.zeros(out_limit)
    right_out_list = np.zeros(out_limit)
    # To prevent go out of the recording range
    check_frames_num = 40

    for num in range(ego_frames.shape[0] - check_frames_num):
        # Keep two array with fixed length and update in every time frame
        left_out_list = left_out_list[-out_limit:]
        right_out_list = right_out_list[-out_limit:]

        # If it is almost the end of recording, don't bother
        # Only run further if the line information exists
        if math.isnan(ego_frames['ego_line_left_begin_x'][num]) or math.isnan(
                ego_frames['ego_line_left_end_x'][num]) or math.isnan(
            ego_frames['ego_line_left_distance_y'][num]) or math.isnan(
            ego_frames['ego_line_left_curv'][num]) or math.isnan(
            ego_frames['ego_line_right_begin_x'][num]) or math.isnan(
            ego_frames['ego_line_right_end_x'][num]) or math.isnan(
            ego_frames['ego_line_right_distance_y'][num]) or math.isnan(ego_frames['ego_line_right_curv'][num]):
            left_out_list = np.append(left_out_list, 0)
            right_out_list = np.append(right_out_list, 0)
            left_out_flag = False
            right_out_flag = False
            continue

        # Detect if the object is out of the ego line area
        ego_line_end = ego_frames['ego_line_left_end_x'][num] if (
                ego_frames['ego_line_left_end_x'][num] > ego_frames['ego_line_right_end_x'][num]) else \
            ego_frames['ego_line_right_end_x'][num]

        # If distance from ego to object is bigger than line end, don't bother
        if ego_line_end <= object_frames['pos_x'][num]:
            left_out_list = np.append(left_out_list, 0)
            right_out_list = np.append(right_out_list, 0)
            left_out_flag = False
            right_out_flag = False
            continue

        # Calculate the y coordinates of left and right line at x position of current object
        left_line_at_obj = - ego_frames['ego_line_left_distance_y'][num] - ego_frames['ego_line_left_curv'][
            num] * ((object_frames['pos_x'][num] - 2) ** 2 / 2)
        right_line_at_obj = - ego_frames['ego_line_right_distance_y'][num] - ego_frames['ego_line_right_curv'][
            num] * ((object_frames['pos_x'][num] - 2) ** 2 / 2)
        # Prepare for cut in monitor, first check if the object is out of the ego line area, minus sign here is because
        # of the default coordinates setting in the recording: left of origin is positive, right is negative
        if -object_frames['pos_y'][num] < left_line_at_obj:
            left_out_flag = True
            right_out_flag = False
            left_out_list = np.append(left_out_list, 1)
            right_out_list = np.append(right_out_list, 0)
            continue
        elif -object_frames['pos_y'][num] > right_line_at_obj:
            right_out_flag = True
            left_out_flag = False
            right_out_list = np.append(right_out_list, 1)
            left_out_list = np.append(left_out_list, 0)
            continue
        else:
            right_out_list = np.append(right_out_list, 0)
            left_out_list = np.append(left_out_list, 0)
            # Counter to check if the next few frames satisfy the cut in condition
            check_frames_num = 40
            # If cut in happens during lane change, don't bother
            start = num - check_frames_num if num - check_frames_num > 0 else 0
            end = num + check_frames_num if num + check_frames_num < ego_frames.shape[0] else ego_frames.shape[0]

            if np.any(lane_change_array[start: end]):
                right_out_flag = False
                left_out_flag = False
                continue

            # Cut in will not labelled if the object stay mostly out of ego line area or the line info doesn't exist
            # for most of time during cut in
            # line_counter = 0
            # for i in range(start+30, end-30):
            #     if math.isnan(ego_frames['ego_line_left_end_x'][i]) or math.isnan(ego_frames['ego_line_right_end_x'][i]):
            #         line_counter = line_counter + 1
            #         continue
            #     else:
            #         ego_line_end = ego_frames['ego_line_left_end_x'][i] if (ego_frames['ego_line_left_end_x'][i] > ego_frames['ego_line_right_end_x'][i]) else ego_frames['ego_line_right_end_x'][i]
            #         if ego_line_end <= object_frames['pos_x'][i]:
            #             line_counter = line_counter + 1
            #         else:
            #             continue
            # print(line_counter)
            # if line_counter > round((end-start-60)/3):
            #     right_out_flag = False
            #     left_out_flag = False
            #     continue

            # If object is within the ego line area,
            # first check the last 20 time frames if the object is out of ego line area at the most time
            if left_out_flag:
                # print('left')
                # print(lane_change_array[num: num + check_frames_num])
                valid_flag = True
                # If a portion(at least half) of last few frames of object lay in the ego line area, don't bother
                if np.sum([left_out_list > 0]) < int(out_limit / 2):
                    valid_flag = False
                else:
                    # Check if the object can stay in the ego line area in the next 40 frames
                    line_exist = 0
                    for i in range(1, check_frames_num):
                        # If in the next 40 frames there are 10 frames have no line information, invalid
                        if line_exist == 10:
                            valid_flag = False
                            break
                        # Check if the line information consistent enough
                        if math.isnan(ego_frames['ego_line_left_begin_x'][num + i]) or math.isnan(
                                ego_frames['ego_line_left_end_x'][num + i]) or math.isnan(
                            ego_frames['ego_line_left_distance_y'][num + i]) or math.isnan(
                            ego_frames['ego_line_left_curv'][num + i]) or math.isnan(
                            ego_frames['ego_line_right_begin_x'][num + i]) or math.isnan(
                            ego_frames['ego_line_right_end_x'][num + i]) or math.isnan(
                            ego_frames['ego_line_right_distance_y'][num + i]) or math.isnan(
                            ego_frames['ego_line_right_curv'][num + i]):
                            line_exist += 1
                            continue
                        else:
                            ego_line_end = ego_frames['ego_line_left_end_x'][i] if (
                                    ego_frames['ego_line_left_end_x'][i] > ego_frames['ego_line_right_end_x'][
                                i]) else ego_frames['ego_line_right_end_x'][i]
                            if ego_line_end <= object_frames['pos_x'][i]:
                                line_exist += 1
                                continue

                        left_line_at_obj = - ego_frames['ego_line_left_distance_y'][num + i] - \
                                           ego_frames['ego_line_left_curv'][num + i] * (
                                                   (object_frames['pos_x'][num + i] - 2) ** 2 / 2)
                        right_line_at_obj = - ego_frames['ego_line_right_distance_y'][num + i] - \
                                            ego_frames['ego_line_right_curv'][num + i] * (
                                                    (object_frames['pos_x'][num + i] - 2) ** 2 / 2)
                        if -object_frames['pos_y'][num + i] < left_line_at_obj or \
                                -object_frames['pos_y'][num + i] > right_line_at_obj:
                            valid_flag = False
                            break
                # Mark the time frame that a complete cut in is finished
                if valid_flag:
                    cut_in_finish_left.append(num)

            if right_out_flag:
                # print('right')
                # print(lane_change_array[num: num + check_frames_num])
                valid_flag = True
                # If a portion(at least half) of last few frames of object lay in the ego line area, don't bother
                if np.sum([right_out_list > 0]) < int(out_limit / 2):
                    valid_flag = False
                else:
                    # Check if the object can stay in the ego line area in the next 40 frames
                    line_exist = 0
                    for i in range(1, check_frames_num):
                        # If in the next 40 frames there are 10 frames have no line information, invalid
                        if line_exist > 15:
                            valid_flag = False
                            break
                        # Check if the line information consistent enough
                        if math.isnan(ego_frames['ego_line_left_begin_x'][num + i]) or math.isnan(
                                ego_frames['ego_line_left_end_x'][num + i]) or math.isnan(
                            ego_frames['ego_line_left_distance_y'][num + i]) or math.isnan(
                            ego_frames['ego_line_left_curv'][num + i]) or math.isnan(
                            ego_frames['ego_line_right_begin_x'][num + i]) or math.isnan(
                            ego_frames['ego_line_right_end_x'][num + i]) or math.isnan(
                            ego_frames['ego_line_right_distance_y'][num + i]) or math.isnan(
                            ego_frames['ego_line_right_curv'][num + i]):
                            line_exist += 1
                            continue
                        else:
                            ego_line_end = ego_frames['ego_line_left_end_x'][num + i] if (
                                    ego_frames['ego_line_left_end_x'][num + i] > ego_frames['ego_line_right_end_x'][
                                num + i]) else ego_frames['ego_line_right_end_x'][num + i]
                            if ego_line_end <= object_frames['pos_x'][num + i]:
                                line_exist += 1
                                continue

                        left_line_at_obj = - ego_frames['ego_line_left_distance_y'][num + i] - \
                                           ego_frames['ego_line_left_curv'][num + i] * (
                                                   (object_frames['pos_x'][num + i] - 2) ** 2 / 2)
                        right_line_at_obj = - ego_frames['ego_line_right_distance_y'][num + i] - \
                                            ego_frames['ego_line_right_curv'][num + i] * (
                                                    (object_frames['pos_x'][num + i] - 2) ** 2 / 2)
                        if -object_frames['pos_y'][num + i] < left_line_at_obj or -object_frames['pos_y'][
                            num + i] > right_line_at_obj:
                            valid_flag = False
                            break
                # Mark the time frame that a complete cut in is finished
                if valid_flag:
                    cut_in_finish_right.append(num)

            right_out_flag = False
            left_out_flag = False

    # 1 second after surrounding vehicle cross the line als cut in
    if len(cut_in_finish_right) > 0:
        right_cut_count = len(cut_in_finish_right)
        for num in cut_in_finish_right:
            cut_in_right_dynamic[num + 25] = 1

    if len(cut_in_finish_left) > 0:
        left_cut_count = len(cut_in_finish_left)
        for num in cut_in_finish_left:
            cut_in_left_dynamic[num + 25] = 1

    return left_cut_count, right_cut_count, cut_in_right_dynamic, cut_in_left_dynamic


def lane_change_detection(ego_df_line):
    """
    Label the lane change
    :param ego_df_line: time frame of ego lane information
    :return: lane_change_left_array: labeled time frame for left lane change
             lane_change_right_array: labeled time frame for right lane change
    """
    m = ego_df_line.shape[0]
    lane_change_left_array = np.zeros(m)
    lane_change_right_array = np.zeros(m)
    left_change = False
    right_change = False
    lane_change_left_list = []
    lane_change_right_list = []
    count = 0
    # # To prevent go out of the recording range
    check_frames_num = 40

    for i in range(ego_df_line.shape[0] - check_frames_num):

        if i % 200 == 0:
            print('Lane change label process {0:.2f}%'.format(i / m * 100))
        # If the ego vehicle touch the line, start to monitor
        if ego_df_line['ego_line_left_distance_y'][i] < 0 and not right_change:
            left_change = True
            right_change = False
        # Monitor process
        if left_change:
            count += 1
            # To prevent the ego did a half lane change and drive back to the previous lane, if in the next 4 seconds
            # there is no lane change detected, reset the monitor
            if count > 100:
                count = 0
                left_change = False
                right_change = False
                continue
            # Lane change condition, count > 5 is to prevent noise
            if count > 5 and ego_df_line['ego_line_left_distance_y'][i] > 2.5:
                count = 0
                left_change = False
                right_change = False
                lane_change_left_list.append(i)
                continue

        if ego_df_line['ego_line_right_distance_y'][i] > 0:
            right_change = True
            left_change = False
        if right_change:
            count += 1
            if count > 100:
                count = 0
                left_change = False
                right_change = False
            if count > 5 and ego_df_line['ego_line_right_distance_y'][i] < -2.5:
                count = 0
                left_change = False
                right_change = False
                lane_change_right_list.append(i)

    # 1 second after ego car across the line, label als lane change
    if len(lane_change_right_list) > 0:
        for i in lane_change_right_list:
            lane_change_right_array[i + 25] = 1
    if len(lane_change_left_list) > 0:
        for i in lane_change_left_list:
            lane_change_left_array[i + 25] = 1
    return lane_change_left_array, lane_change_right_array


def label(ego_df, objects_static_df, objects_dynamic_df, file_number):
    """
    Main function for labelling
    :param ego_df: the whole ego time frame
    :param objects_static_df: the objects statistic time frame of ego
    :param objects_dynamic_df: the objects dynamic time frame of ego
    :param file_number: to track how many files have been processed
    :return: ego_df: the ego time frame with lane change label columns
             objects_static_df: the objects static time frame with cut in count
             objects_dynamic_df: the objects dynamic time frame with cut in label columns
    """
    grouped_dynamic = objects_dynamic_df.groupby(['obj_id'], sort=False)
    cut_in_left_count_list = []
    cut_in_right_count_list = []
    cut_in_dynamic_left_list = []
    cut_in_dynamic_right_list = []
    obj_num = objects_static_df.shape[0]

    # Get lane change label
    ego_df_line = ego_df.loc[:, ['ego_line_left_distance_y', 'ego_line_right_distance_y']]
    lane_change_left_list, lane_change_right_list = lane_change_detection(ego_df_line)

    # Construct a array that show when lane change happens, later during lane change, cut in is not allow to happen
    lane_change_array = lane_change_conflict(np.array(lane_change_left_list) + np.array(lane_change_right_list))

    for obj_id, obj_rows in grouped_dynamic:
        print('Cut in label the {}.data process: {:.2f}%'.format(str(file_number + 1), obj_id / obj_num * 100))
        # The first record in df start from index 0
        initial_frame = objects_static_df['initial_frame'][obj_id - 1]
        total_frames = objects_static_df['total_frames'][obj_id - 1]

        # Unlike python slicing, here both the start and the stop are included
        ego_frames = ego_df.loc[initial_frame:initial_frame + total_frames - 1, ['ego_line_left_begin_x',
                                                                                 'ego_line_left_end_x',
                                                                                 'ego_line_left_distance_y',
                                                                                 'ego_line_left_curv',
                                                                                 'ego_line_right_begin_x',
                                                                                 'ego_line_right_end_x',
                                                                                 'ego_line_right_distance_y',
                                                                                 'ego_line_right_curv']]
        ego_frames.reset_index(drop=True, inplace=True)
        obj_rows.reset_index(drop=True, inplace=True)

        # Construct a array that show when lane change happens, later during lane change, cut in is not allow to happen
        lane_change_sub_array = lane_change_array[initial_frame:initial_frame + total_frames]

        left_cut_count, right_cut_count, cut_in_right_dynamic, cut_in_left_dynamic = cut_in_detection(ego_frames,
                                                                                                      obj_rows,
                                                                                                      lane_change_sub_array)
        cut_in_left_count_list.append(left_cut_count)
        cut_in_right_count_list.append(right_cut_count)
        cut_in_dynamic_left_list = [*cut_in_dynamic_left_list, *cut_in_left_dynamic]
        cut_in_dynamic_right_list = [*cut_in_dynamic_right_list, *cut_in_right_dynamic]

    objects_static_df['cut_in_left'] = cut_in_left_count_list
    objects_static_df['cut_in_right'] = cut_in_right_count_list
    objects_dynamic_df['cut_in_left'] = cut_in_dynamic_left_list
    objects_dynamic_df['cut_in_right'] = cut_in_dynamic_right_list
    ego_df['lane_change_left'] = lane_change_left_list
    ego_df['lane_change_right'] = lane_change_right_list

    return ego_df, objects_static_df, objects_dynamic_df
