import os
import pandas as pd
from NN_preprocess_utils import *

if __name__ == "__main__":

    path = r'C:\Users\A6OJTFD\Desktop\concatenate_data'
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith("ego.csv"):
                file_list.append(os.path.join(root, file))

    # Depend on number of features and length of each example
    feature_num = 5
    example_length = 500
    # Calculated based on Conv1D layer
    label_length = int((example_length-7))/2 + 1
    X = np.zeros((example_length, feature_num))
    Y = np.zeros((label_length, 1))

    for sub_path in file_list:
        x, y = construct_feature(sub_path)
        X = np.concatenate((X, x), axis=0)
        Y = np.concatenate((Y, y), axis=0)
        X, mean, std = normalization(X)
        X, Y = fit_dims(X, Y)

    np.save('', X)
    np.save('', Y)
    np.save('', np.concatenate((mean, std), axis=0))


