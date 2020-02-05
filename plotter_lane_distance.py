import os
# import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

if __name__ == "__main__":

    path = r'processed_data\csv\first\0_cv_ego.csv'
    # Walk through every ego data

    # Depend on number of features and length of each example
    start_frame = 10000
    window_size = 5000

    columns_name = ['ego_line_left_distance_y']

    # columns_name = ['speed', 'acc_x', 'acc_y']
    # trained_model = load_model(r'NN_data\model_without_steering_ang_6.h5')

    # mean_and_variance = np.load(r'NN_data\mean_std.npy')

    df = pd.read_csv(path)
    sample = df[columns_name][start_frame:start_frame+window_size]

    # normalized_sample = (sample - mean_and_variance[0, :-1])/mean_and_variance[1, :-1]
    # plt.figure(figsize=(10, 100))
    # plt.plot(sample)
    ax = plt.subplot(111)
    ax.plot(sample, lw=2.5, color=(31/255, 119/255, 180/255))
    ax.set_title('distance to left ego line')

    ax.spines["top"].set_visible(False)
    # ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # ax.spines["left"].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    # fig = plt.figure()
    # axes_1 = fig.add_plot()
    # axes_1.plot(y_predicted[0, :, 0])
    # axes_1.set_title('Free driving label - Predicted')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '%g' % (x / 25 - 400)))
    # axes_1.set_ylim([-0.1, 1.1])
    plt.xlabel('time frame (s)')
    plt.ylabel('distance (m)')
    # plt.title('distance to left ego line')

    plt.show()