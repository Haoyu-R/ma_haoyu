import os
import pandas as pd


# Given a path, concatenate the csv files to on in every sub directory
directory_path = r'C:\Users\A6OJTFD\Desktop\concatenate_data'
sub_directory_path = [x[0] for x in os.walk(directory_path)]

idx = 0
for path in sub_directory_path[1:]:
    print(path)
    file_list = []
    idx = idx + 1
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".csv"):
                file_list.append(os.path.join(root, file))
                print(file)
    combined_csv = pd.concat([pd.read_csv(f) for f in file_list], sort=False)
    combined_csv.to_csv(r"{}\combined_{}.csv".format(directory_path, idx), index=False, encoding='utf-8-sig')