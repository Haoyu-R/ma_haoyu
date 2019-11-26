import os
import glob
import pandas as pd


def if_steering_angle(df):
    """
    Judge if there is steering angle info in the data
    :param df: raw data
    :return: True/False
    """
    steering_ang_column = [col for col in df.columns if 'LWI_Lenkradwinkel' in col]
    if len(steering_ang_column) == 1:
        return True
    else:
        return False


if __name__ == '__main__':
    path = r'C:\Users\A6OJTFD\Desktop\allTraces\pdwG'
    if os.path.isdir(path):
        csv_files_path = glob.glob(path + r'\*.csv')
        judge_list = []
        for index, csv_path in enumerate(csv_files_path):
            csv_df = pd.read_csv(csv_path)
            judge_list.append(if_steering_angle())
            print(if_steering_angle(csv_df))




