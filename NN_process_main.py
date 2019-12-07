from mdm_data_utils import *

if __name__ == r"__main__":
    path = r'..\concatenate_data\not_processed_1'
    csv_files_path = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".csv"):
                csv_files_path.append(os.path.join(root, file))


    check_valid_index = True
    for index, csv_path in enumerate(csv_files_path):
        index = index + 31
        print(csv_path)

        print('Loading the {}. data...'.format(str(index + 1)))
        csv_df = pd.read_csv(csv_path)
        first_valid_index, last_valid_index, not_valid, error = frame_count(csv_df, check_valid_index)
        if not_valid:
            print('{}: '.format(csv_path)+error)
            continue
        ego_df, if_steering_ang = new_ego_df(csv_df, first_valid_index, last_valid_index)
        objects_static_df, objects_dynamic_df, is_SDF = new_objects_df(csv_df, first_valid_index,
                                                                       last_valid_index,
                                                                       index, 30)
        ego_df, objects_static, objects_dynamic = label(ego_df, objects_static_df, objects_dynamic_df,
                                                        index)
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

        # Path 1
        ego_df.to_csv(r"..\preprocessed_data\\" + str(index) + last_str_ego, index=False)
        objects_static.to_csv(r"..\preprocessed_data\\" + str(index) + last_str_static,
                              index=False)
        objects_dynamic.to_csv(r"..\preprocessed_data\\" + str(index) + last_str_dynamic,
                               index=False)

        # Path 2
        # ego_df.to_csv(r"..\ma_haoyu\processed_data\csv\\" + str(index) + last_str_ego, index=False)
        # objects_static.to_csv(r"..\ma_haoyu\processed_data\csv\\" + str(index) + last_str_static,
        #                       index=False)
        # objects_dynamic.to_csv(r"..\processed_data\csv\\" + str(index) + last_str_dynamic,
        #                        index=False)

    print('All csv files in the folder processed')


