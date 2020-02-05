from mdm_data_utils import *

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
            #
            counter = 0
            cs = []
            name = str(os.path.basename(csv_path))
            total_chunk = 295
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

    print('All csv files in the folder processed')
