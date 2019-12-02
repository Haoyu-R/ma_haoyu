import os
import pandas as pd

path = r'C:\Users\A6OJTFD\Desktop\allTraces\extractTestdala_2019_05_21_e7ce2851-e2ca-4805-b4bb-bbfeeff558b2_Data2.clf.csv'

file_list = []
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".csv"):
            file_list.append(os.path.join(root, file))

combined_csv = pd.concat([pd.read_csv(f) for f in file_list])
combined_csv.to_csv(r"C:\Users\A6OJTFD\Desktop\allTraces\extractTestdala_2019_05_21_e7ce2851-e2ca-4805-b4bb-bbfeeff558b2_Data2.clf.csv\combined_csv.csv", index=False, encoding='utf-8-sig')