from mdm_data_utils import *
import dask.dataframe as dd

if __name__ == r"__main__":
    path = r'..\concatenate_data\too_big'
    csv_files_path = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".csv"):
                csv_files_path.append(os.path.join(root, file))

    for csv_path in csv_files_path:

        with open(csv_path, "r") as traj_file:

            # iter_csv = pd.read_csv(traj_file, delimiter=",", header=0, chunksize=10000, dtype='float')
            # total_chunk = 0
            # for j in iter_csv:
            #     total_chunk += 1
            # print("total_chunks: " + str(total_chunk))
            # iter_csv = 0

            counter = 0
            cs = []
            name = str(os.path.basename(csv_path))
            total_chunk = 382
            for j in pd.read_csv(traj_file, delimiter=",", header=0, chunksize=10000, dtype='float'):
                counter += 1
                print(counter)
                cs.append(j)

                if int(1 / 10 * total_chunk) == counter:
                    df = pd.concat(cs)

                    df.to_csv(r"..\concatenate_data\not_processed_1\\" + "0_" + name,
                              index=False)
                    cs = []
                if int(2 / 10 * total_chunk) == counter:
                    df = pd.concat(cs)
                    df.to_csv(r"..\concatenate_data\not_processed_1\\" + "1_" + name,
                              index=False)
                    cs = []
                if int(3 / 10 * total_chunk) == counter:
                    df = pd.concat(cs)
                    df.to_csv(r"..\concatenate_data\not_processed_1\\" + "2_" + name,
                              index=False)
                    cs = []
                if int(4 / 10 * total_chunk) == counter:
                    df = pd.concat(cs)
                    df.to_csv(r"..\concatenate_data\not_processed_1\\" + "3_" + name,
                              index=False)
                    cs = []
                if int(5 / 10 * total_chunk) == counter:
                    df = pd.concat(cs)

                    df.to_csv(r"..\concatenate_data\not_processed_1\\" + "4_" + name,
                              index=False)
                    cs = []
                if int(6 / 10 * total_chunk) == counter:
                    df = pd.concat(cs)
                    df.to_csv(r"..\concatenate_data\not_processed_1\\" + "5_" + name,
                              index=False)
                    cs = []
                if int(7 / 10 * total_chunk) == counter:
                    df = pd.concat(cs)
                    df.to_csv(r"..\concatenate_data\not_processed_1\\" + "6_" + name,
                              index=False)
                    cs = []
                if int(8 / 10 * total_chunk) == counter:
                    df = pd.concat(cs)
                    df.to_csv(r"..\concatenate_data\not_processed_1\\" + "7_" + name,
                              index=False)
                    cs = []
                if int(9 / 10 * total_chunk) == counter:
                    df = pd.concat(cs)
                    df.to_csv(r"..\concatenate_data\not_processed_1\\" + "8_" + name,
                              index=False)
                    cs = []
                if int(10 / 10 * total_chunk) == counter:
                    df = pd.concat(cs)
                    df.to_csv(r"..\concatenate_data\not_processed_1\\" + "9_" + name,
                              index=False)
                    cs = []

    path = r'..\concatenate_data\not_processed_1'
    csv_files_path = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".csv"):
                csv_files_path.append(os.path.join(root, file))

    check_valid_index = True
    for index, csv_path in enumerate(csv_files_path):
        index = index + 31
        print('Loading the {}. data...'.format(str(index + 1)))
        csv_df = pd.read_csv(csv_path)
        first_valid_index, last_valid_index, not_valid, error = frame_count(csv_df, check_valid_index)
        if not_valid:
            print('{}: '.format(csv_path) + error)
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
    #
    # check_valid_index = True
    # for index, csv_path in enumerate(csv_files_path):
    #     index = index + 31
    #     print('Loading the {}. data...'.format(str(index + 1)))
    #     # cs = []
    #     with open(csv_path, "r") as traj_file:
    #         # iter_csv = pd.read_csv(traj_file, delimiter=",", header=0, chunksize=10000, dtype='float')
    #         # total_chunk = 0
    #         # for j in iter_csv:
    #         #     total_chunk += 1
    #         # print("total_chunks: " + str(total_chunk))
    #         total_chunk = 295
    #         counter = 0
    #         cs = []
    #         name = str(os.path.basename(csv_path))
    #         for j in pd.read_csv(traj_file, delimiter=",", header=0, chunksize=10000, dtype='float'):
    #             counter += 1
    #             print(counter)
    #             cs.append(j)
    #             # if int(1 / 4 * total_chunk) == counter:
    #             if int(1 / 4 * total_chunk) == counter:
    #                 df = pd.concat(cs)
    #
    #                 df.to_csv(r"..\concatenate_data\not_processed_1\\"  + "0_" + name,
    #                           index=False)
    #                 cs = []
    #             if int(2 / 4 * total_chunk) == counter:
    #                 df = pd.concat(cs)
    #                 df.to_csv(r"..\concatenate_data\not_processed_1\\" + "1_" + name,
    #                           index=False)
    #                 cs = []
    #             if int(3 / 4 * total_chunk) == counter:
    #                 df = pd.concat(cs)
    #                 df.to_csv(r"..\concatenate_data\not_processed_1\\" + "2_" + name,
    #                           index=False)
    #                 cs = []
    #             if int(4 / 4 * total_chunk) == counter:
    #                 df = pd.concat(cs)
    #                 df.to_csv(r"..\concatenate_data\not_processed_1\\" + "3_" + name,
    #                           index=False)
    #                 cs = []

            # for i in range(4):
            #     cs = []
            #     print(i)
            #     for idx, j in enumerate(iter_csv):
            #         if int((i + 1) / 4 * total_chunk) > idx >= int(i / 4 * total_chunk):
            #             cs.append(j)
            #             print(idx)
            #     df = pd.concat(cs)
            #     df.to_csv(r"..\concatenate_data\not_processed_1\\" + str(i) + "+" + os.path.basename(traj_file),
            #               index=False)

        # csv_df = pd.read_csv(csv_path, dtype='float64')
        # first_valid_index, last_valid_index, not_valid, error = frame_count(csv_df, check_valid_index)
        # if not_valid:
        #     print('{}: '.format(csv_path)+error)
        #     continue
        # ego_df, if_steering_ang = new_ego_df(csv_df, first_valid_index, last_valid_index)
        # objects_static_df, objects_dynamic_df, is_SDF = new_objects_df(csv_df, first_valid_index,
        #                                                                last_valid_index,
        #                                                                index, 30)
        # ego_df, objects_static, objects_dynamic = label(ego_df, objects_static_df, objects_dynamic_df,
        #                                                 index)
        # # Name the data based on steering ang info/ sdf source/ cv source
        # last_str_ego = ''
        # last_str_static = ''
        # last_str_dynamic = ''
        # if if_steering_ang:
        #     if is_SDF:
        #         last_str_ego = '_sdf_ego.csv'
        #         last_str_static = '_sdf_static.csv'
        #         last_str_dynamic = '_sdf_dynamic.csv'
        #     else:
        #         last_str_ego = '_cv_ego.csv'
        #         last_str_static = '_cv_static.csv'
        #         last_str_dynamic = '_cv_dynamic.csv'
        # else:
        #     if is_SDF:
        #         last_str_ego = '_no_steering_sdf_ego.csv'
        #         last_str_static = '_no_steering_sdf_static.csv'
        #         last_str_dynamic = '_no_steering_sdf_dynamic.csv'
        #     else:
        #         last_str_ego = '_no_steering_cv_ego.csv'
        #         last_str_static = '_no_steering_cv_static.csv'
        #         last_str_dynamic = '_no_steering_cv_dynamic.csv'
        #
        # # Path 1
        # ego_df.to_csv(r"..\preprocessed_data\\" + str(index) + last_str_ego, index=False)
        # objects_static.to_csv(r"..\preprocessed_data\\" + str(index) + last_str_static,
        #                       index=False)
        # objects_dynamic.to_csv(r"..\preprocessed_data\\" + str(index) + last_str_dynamic,
        #                        index=False)

        # Path 2
        # ego_df.to_csv(r"..\ma_haoyu\processed_data\csv\\" + str(index) + last_str_ego, index=False)
        # objects_static.to_csv(r"..\ma_haoyu\processed_data\csv\\" + str(index) + last_str_static,
        #                       index=False)
        # objects_dynamic.to_csv(r"..\processed_data\csv\\" + str(index) + last_str_dynamic,
        #                        index=False)
    # print('All csv files in the folder processed')
