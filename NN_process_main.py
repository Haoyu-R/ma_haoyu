import numpy as np
import pandas as pd
import os
from NN_preprocess_utils import *

if __name__ == r"__main__":
    path = r'C:\Users\arhyr\Desktop\audi\ma_haoyu'
    sample_list = []
    timestamp_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith("ego.csv"):
                # print(os.path.join(root, file))
                sample_dfs, y = sample_in_file(os.path.join(root, file))
                sample_list.append(sample_dfs)
                timestamp_list.append(y)
    normalized_samples = sample_norm(sample_list)
    Y = get_label(timestamp_list)


