import tensorflow as tf
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

# Deactivate the GPU
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# Clear back sessions
tf.keras.backend.clear_session()

X = np.load(r'..\preprocessed_data\test_with_steering_angle\X_all.npy')
Y = np.load(r'..\preprocessed_data\test_with_steering_angle\Y_all.npy')

trained_model = load_model(r'NN_data\model_20_with_steering_both.h5')
num_ = 400

# Following visualize the label in on sample
for i in range(100):
    num = num_+i
    y_test = trained_model.predict(np.expand_dims(X[num, :, :], axis=0))

    fig = plt.figure()

    axes_1 = fig.add_subplot(2, 5, 1)
    axes_1.plot(y_test[0, :, 0])
    axes_1.set_title('Free driving label - Predicted')
    axes_1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '%g' % (x * 0.08)))
    axes_1.set_ylim([-0.1, 1.1])

    axes_2 = fig.add_subplot(2, 5, 2)
    axes_2.plot(y_test[0, :, 1])
    axes_2.set_title('Left lane change label - Predicted')
    axes_2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '%g' % (x * 0.08)))
    axes_2.set_ylim([-0.1, 1.1])

    axes_3 = fig.add_subplot(2, 5, 3)
    axes_3.plot(y_test[0, :, 2])
    axes_3.set_title('Right lane change label - Predicted')
    axes_3.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '%g' % (x * 0.08)))
    axes_3.set_ylim([-0.1, 1.1])

    axes_4 = fig.add_subplot(2, 5, 4)
    axes_4.plot(y_test[0, :, 3])
    axes_4.set_title('Left cut in label - Predicted')
    axes_4.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '%g' % (x * 0.08)))
    axes_4.set_ylim([-0.1, 1.1])

    axes_5 = fig.add_subplot(2, 5, 5)
    axes_5.plot(y_test[0, :, 4])
    axes_5.set_title('Right cut in label - Predicted')
    axes_5.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '%g' % (x * 0.08)))
    axes_5.set_ylim([-0.1, 1.1])

    axes_6 = fig.add_subplot(2, 5, 6)
    axes_6.plot(Y[num, :, 0])
    axes_6.set_title('Free driving label - Reference')
    axes_6.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '%g' % (x * 0.08)))
    axes_6.set_ylim([-0.1, 1.1])

    axes_7 = fig.add_subplot(2, 5, 7)
    axes_7.plot(Y[num, :, 1])
    axes_7.set_title('Left lane change label - Reference')
    axes_7.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '%g' % (x * 0.08)))
    axes_7.set_ylim([-0.1, 1.1])

    axes_8 = fig.add_subplot(2, 5, 8)
    axes_8.plot(Y[num, :, 2])
    axes_8.set_title('Right lane change  label - Reference')
    axes_8.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '%g' % (x * 0.08)))
    axes_8.set_ylim([-0.1, 1.1])

    axes_9 = fig.add_subplot(2, 5, 9)
    axes_9.plot(Y[num, :, 3])
    axes_9.set_title('Left cut in label - Reference')
    axes_9.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '%g' % (x * 0.08)))
    axes_9.set_ylim([-0.1, 1.1])

    axes_10 = fig.add_subplot(2, 5, 10)
    axes_10.plot(Y[num, :, 4])
    axes_10.set_title('Right cut in label - Reference')
    axes_10.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '%g' % (x * 0.08)))
    axes_10.set_ylim([-0.1, 1.1])

    plt.show()