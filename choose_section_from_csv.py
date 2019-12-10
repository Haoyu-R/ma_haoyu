import numpy as np
import pandas as pd
from pathlib import Path
import os


def find_index(df, start_day, start_hour, start_min, start_sec,
               end_day, end_hour, end_min, end_sec):
    m = df.shape[0]

    day_col = [loc for loc in df.columns if 'Tag' in loc]
    hour_col = [loc for loc in df.columns if 'Stunde' in loc]
    min_col = [loc for loc in df.columns if 'Minute' in loc]
    sec_col = [loc for loc in df.columns if 'Sekunde' in loc]

    start_frm = np.nan
    end_frm = np.nan

    found_start = False
    found_end = False

    for i in range(m):
        if i % 200 == 0:
            print('Processed: {:.2f}%'.format(i/m*100))
        day = df[day_col[0]][i]
        hour = df[hour_col[0]][i]
        min = df[min_col[0]][i]
        sec = df[sec_col[0]][i]
        if not np.isnan(day) and not np.isnan(hour) and not np.isnan(min) and not np.isnan(sec):
            day = int(day)
            hour = int(hour)
            min = int(min)
            sec = int(sec)
            if not found_start and day == start_day and hour == start_hour and min == start_min and sec == start_sec:
                found_start = True
                start_frm = i
            if not found_end and day == end_day and hour == end_hour and min == end_min and sec == end_sec:
                found_end = True
                end_frm = i
                print('Start and end index found')
                break

    if np.isnan(start_frm) or np.isnan(end_frm):
        print('Did not find the correspondent df')
        exit(1)
    else:
        return start_frm, end_frm


def zero_pad(string):
    num = int(string)
    if num < 10:
        return '0' + str(num)
    else:
        return str(num)


if __name__ == '__main__':
    path = r'C:\Users\A6OJTFD\Desktop\MDM data process\row_data_cut\Trace_e-tron_10_29.csv'
    start_day = 29
    start_hour = 13
    start_min = 32
    start_second = 9
    end_day = 29
    end_hour = 14
    end_min = 15
    end_second = 24

    if os.path.isfile(path):
        dir_name = os.path.dirname(path)
        file_name = Path(path).stem
        print('Loading the data...')
        df = pd.read_csv(path)
        start_frame, end_frame = find_index(df, start_day, start_hour, start_min, start_second,
                                            end_day, end_hour, end_min, end_second)
        chunk = df.loc[start_frame:end_frame]

        chunk.to_csv(dir_name + '\processed_data\\' + 'chunk_city' + zero_pad(start_day) + '_' + zero_pad(start_hour) + zero_pad(start_min) +
                     zero_pad(start_second) + '_' + zero_pad(end_hour) + zero_pad(end_min) + zero_pad(end_second) + '.csv', index=False)
