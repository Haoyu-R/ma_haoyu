import pandas as pd
import numpy as np
import os
from pathlib import Path


def cut_in_chunk(df):
    rows_num = df.shape[0]
    chunk_size = rows_num // 50
    chunk_indices = list(range(chunk_size, (50 + 1) * chunk_size, chunk_size))
    chunks = np.split(df, chunk_indices)
    return chunks


if __name__ == "__main__":
    path = r'C:\Users\A6OJTFD\Desktop\allTraces\pdwG\Trace_e-tron_10_29.csv'
    dir_path = os.path.dirname(path)
    file_name = Path(path).stem
    if os.path.isfile(path):
        df = pd.read_csv(path)
        chunks = cut_in_chunk(df)
        for index, chunk in enumerate(chunks):
            print('Progress rate: {}%'.format(index/50*100))
            chunk.to_csv(dir_path + '\\' + 'part-' + str(index) + '-' + file_name + '.csv', index=False)


