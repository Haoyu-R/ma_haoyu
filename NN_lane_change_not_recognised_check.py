import os
# import tensorflow as tf
from keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    path = r'processed_data\csv\first\0_cv_ego.csv'
    # Walk through every ego data

    # Depend on number of features and length of each example
# <<<<<<< HEAD
#     start_frame = 18700
#     window_size = 500
#
#     # columns_name = ['speed', 'acc_x', 'acc_y', 'steering_ang']
#     # trained_model = load_model(r'NN_data\my_model.h5')
# =======
#     start_frame = 10000
#     window_size = 10000
#
#     columns_name = ['speed', 'acc_x', 'acc_y']
#     trained_model = load_model(r'NN_data\model_12_without_steering_LSTM_bi.h5')
# >>>>>>> 8a8a09d237282c56f77286390fadbd3946ebb716
#
#     columns_name = ['acc_x', 'acc_y', 'steering_ang']
#     trained_model = load_model(r'NN_data\model_without_steering_ang_6.h5')
#
#     mean_and_variance = np.load(r'NN_data\mean_std.npy')
#
#     df = pd.read_csv(path)
#     sample = df[columns_name][start_frame: start_frame+window_size]
#
# <<<<<<< HEAD
#     normalized_sample = (sample - mean_and_variance[0, 1:])/mean_and_variance[1, 1:]
#     # normalized_sample = (sample - mean_and_variance[0, :]) / mean_and_variance[1, :]
# =======
#     # normalized_sample = (sample - mean_and_variance[0, :-1])/mean_and_variance[1, :-1]
#     normalized_sample = (sample - mean_and_variance[0, :-1]) / mean_and_variance[1, :-1]
# >>>>>>> 8a8a09d237282c56f77286390fadbd3946ebb716


    # Following visualize the label in on sample
    y_predicted = trained_model.predict(np.expand_dims(normalized_sample, axis=0))

    fig = plt.figure()
    axes_1 = fig.add_subplot(3, 1, 1)
    axes_1.plot(y_predicted[0, :, 0])
    axes_1.set_title('Free driving label - Predicted')
    axes_1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '%g' % (x * 2)))
    axes_1.set_ylim([-0.1, 1.1])
    # axes_1.set_autoscaley_on(False)

    axes_2 = fig.add_subplot(3, 1, 2)
    axes_2.plot(y_predicted[0, :, 1])
    axes_2.set_title('Left lane change label - Predicted')
    axes_2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '%g' % (x * 2)))
    axes_2.set_ylim([-0.1, 1.1])

    axes_3 = fig.add_subplot(3, 1, 3)
    axes_3.plot(y_predicted[0, :, 2])
    axes_3.set_title('Right lane change label - Predicted')
    axes_3.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '%g' % (x * 2)))
    axes_3.set_ylim([-0.1, 1.1])
    plt.tight_layout()
    plt.show()

