# Haoyu Ren
# 11.11.2019
# Visualization of data

import os
import sys
import cv2
import pandas as pd
from visualization_utils import *


def read_files(file_name):
    """
    Read the ego, dynamic and static file for one recording
    :param file_number: the number in current file name
    :return: three dataframe contain ego, dynamic and static respectively
    """
    if os.path.exists(
            r'..\MDM data process\processed_data\csv\{}_dynamic.csv'.format(file_name)) and os.path.exists(
        r'..\MDM data process\processed_data\csv\{}_static.csv'.format(file_name)) and os.path.exists(
        r'..\MDM data process\processed_data\csv\{}_ego.csv'.format(file_name)):
        with open(r'..\MDM data process\processed_data\csv\{}_dynamic.csv'.format(file_name)) as tmp_dynamic:
            dynamic_csv = pd.read_csv(tmp_dynamic)
            print('Dynamic csv file found')
        with open(r'..\MDM data process\processed_data\csv\{}_static.csv'.format(file_name)) as tmp_static:
            static_csv = pd.read_csv(tmp_static)
            print('Static csv file found')
        with open(r'..\MDM data process\processed_data\csv\{}_ego.csv'.format(file_name)) as tmp_ego:
            ego_csv = pd.read_csv(tmp_ego)
            print('Ego csv file found')
        return ego_csv, dynamic_csv, static_csv
    else:
        print('No available data')
        sys.exit(0)


if __name__ == "__main__":
    ego, dynamic_raw, static_raw = read_files('0_cv')
    # ego, dynamic_raw, static_raw = read_files('one')
    # try:
    dynamic = process_dynamic(dynamic_raw)
    static = process_static(static_raw)

    play_video = True

    try:
        if dynamic is not None and static is not None and ego is not None:
            if play_video:
                frame_count, cap = video(r'..\MDM data process\video\processed_video\output.avi')
                if frame_count > 0:
                    visualization_plot = VisualizationPlot(ego, dynamic, static, play_video, frame_count, cap)
                    visualization_plot.show()
                else:
                    print('Video file damaged')
            else:
                visualization_plot = VisualizationPlot(ego, dynamic, static, play_video)
                visualization_plot.show()
    except:
        print('Something went wrong when process the raw data')
        sys.exit(1)
