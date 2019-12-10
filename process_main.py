# Haoyu Ren
# 11.11.2019
# Process to new data structure, label cut in and lane change

from mdm_data_utils import *

if __name__ == "__main__":

    # Path of raw data and path for saving of new data
    read_path = r'..\MDM data process\raw_data\csv'
    save_path = r"..\MDM data process\processed_data\csv\second\\"
    # read_path = r'..\ma_haoyu\raw_data\csv'
    # save_path = r"..\ma_haoyu\processed_data\csv\\"

    # If need to run index check, be careful if want to synchronize with video, do not set to True
    check_valid_index = False
    if os.path.isdir(read_path):
        csv_files_path = glob.glob(read_path + r'\*.csv')
        for index, csv_path in enumerate(csv_files_path):
            print('Loading the {}. data...'.format(str(index+1)))
            csv_df = pd.read_csv(csv_path)
            first_valid_index, last_valid_index, not_valid, info = frame_count(csv_df, check_valid_index)
            if not_valid:
                print('{}: {}'.format(info, csv_path))
                continue
            ego_df, if_steering_ang = new_ego_df(csv_df, first_valid_index, last_valid_index)
            objects_static_df, objects_dynamic_df, is_SDF = new_objects_df(csv_df, first_valid_index, last_valid_index,
                                                                           index, 30)
            ego_df, objects_static, objects_dynamic = label(ego_df, objects_static_df, objects_dynamic_df, index)
            # Name the data based on steering ang info/ sdf source/ cv source
            last_str_ego = ''
            last_str_static = ''
            last_str_dynamic = ''
            if if_steering_ang:
                if is_SDF:
                    last_str_ego = '_sdf_ego.csv'
                    last_str_static = '_sdf_static.csv'
                    last_str_dynamic = '_sdf_dynamic.csv'
                else:
                    last_str_ego = '_cv_ego.csv'
                    last_str_static = '_cv_static.csv'
                    last_str_dynamic = '_cv_dynamic.csv'
            else:
                if is_SDF:
                    last_str_ego = '_no_steering_sdf_ego.csv'
                    last_str_static = '_no_steering_sdf_static.csv'
                    last_str_dynamic = '_no_steering_sdf_dynamic.csv'
                else:
                    last_str_ego = '_no_steering_cv_ego.csv'
                    last_str_static = '_no_steering_cv_static.csv'
                    last_str_dynamic = '_no_steering_cv_dynamic.csv'

            ego_df.to_csv("{}".format(save_path) + str(index) + last_str_ego, index=False)
            objects_static.to_csv("{}".format(save_path) + str(index) + last_str_static,
                                  index=False)
            objects_dynamic.to_csv("{}".format(save_path) + str(index) + last_str_dynamic,
                                   index=False)

        print('All csv files in the folder processed')

    elif os.path.isfile(read_path):
        print('Loading the data...')
        csv_df = pd.read_csv(read_path)
        first_valid_index, last_valid_index, not_valid, info = frame_count(csv_df, check_valid_index)
        if not_valid:
            print('{}: {}'.format(info, read_path))
        else:
            ego_df, if_steering_ang = new_ego_df(csv_df, first_valid_index, last_valid_index)
            objects_static_df, objects_dynamic_df, is_SDF = new_objects_df(csv_df, first_valid_index, last_valid_index, 20)
            ego_df, objects_static, objects_dynamic = label(ego_df, objects_static_df, objects_dynamic_df)

            # Name the data based on steering ang info/ sdf source/ cv source
            last_str_ego = ''
            last_str_static = ''
            last_str_dynamic = ''
            if if_steering_ang:
                if is_SDF:
                    last_str_ego = '_sdf_ego.csv'
                    last_str_static = '_sdf_static.csv'
                    last_str_dynamic = '_sdf_dynamic.csv'
                else:
                    last_str_ego = '_cv_ego.csv'
                    last_str_static = '_cv_static.csv'
                    last_str_dynamic = '_cv_dynamic.csv'
            else:
                if is_SDF:
                    last_str_ego = '_no_steering_sdf_ego.csv'
                    last_str_static = '_no_steering_sdf_static.csv'
                    last_str_dynamic = '_no_steering_sdf_dynamic.csv'
                else:
                    last_str_ego = '_no_steering_cv_ego.csv'
                    last_str_static = '_no_steering_cv_static.csv'
                    last_str_dynamic = '_no_steering_cv_dynamic.csv'

            ego_df.to_csv("{}".format(save_path) + 'one' + last_str_ego, index=False)
            objects_static.to_csv("{}".format(save_path) + 'one'+ last_str_static,
                                  index=False)
            objects_dynamic.to_csv("{}".format(save_path) + 'one' + last_str_dynamic,
                                   index=False)
            print('csv file processed')
    else:
        print(r'Didn\'t find any csv file')
