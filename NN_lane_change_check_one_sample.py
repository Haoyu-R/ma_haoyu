import tensorflow as tf
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
from sklearn.metrics import classification_report


# Deactivate the GPU
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"


# Clear back sessions
tf.keras.backend.clear_session()

# Used to select the best learning rate
# lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10 ** (epoch/20))

X = np.load(r'NN_data\X.npy')
Y = np.load(r'NN_data\Y.npy')

# X = np.load(r'NN_data\X.npy')
# Y = np.load(r'NN_data\Y.npy')

# <<<<<<< HEAD
# trained_model = load_model(r'NN_data\model_without_steering_ang_6.h5')
# # trained_model = load_model(r'NN_data\my_model.h5')
# num_ = 1000
# # Following visualize the label in on sample
# for i in range(10):
#     num = num_ + i
# =======
trained_model = load_model(r'NN_data\model_without_steering_ang_8.h5')
num_ = 1050
# Following visualize the label in on sample
for i in range(100):
    num = num_+i
# >>>>>>> 8a8a09d237282c56f77286390fadbd3946ebb716
    y_test = trained_model.predict(np.expand_dims(X[num, :, :-1], axis=0))

    fig = plt.figure()

    # axes_4 = fig.add_subplot(3, 1, 1)
    # axes_4.plot(Y[num, :, 0])
    # axes_4.set_title('Free driving label')
    # axes_4.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '%g' % (x * 0.08)))
    # axes_4.set_ylim([-0.1, 1.1])
    # axes_4.set_xlabel("time (s)")
    #
    # axes_5 = fig.add_subplot(3, 1, 2)
    # axes_5.plot(Y[num, :, 1])
    # axes_5.set_title('Left lane change label')
    # axes_5.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '%g' % (x * 0.08)))
    # axes_5.set_ylim([-0.1, 1.1])
    # axes_5.set_xlabel("time (s)")
    #
    # axes_6 = fig.add_subplot(3, 1, 3)
    # axes_6.plot(Y[num, :, 2])
    # axes_6.set_title('Right lane change label')
    # axes_6.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '%g' % (x * 0.08)))
    # axes_6.set_ylim([-0.1, 1.1])
    # axes_6.set_xlabel("time (s)")
    # plt.tight_layout()
    axes_1 = fig.add_subplot(2, 3, 1)
    axes_1.plot(y_test[0, :, 0])
    axes_1.set_title('Free driving label - Predicted')
    axes_1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '%g' % (x * 0.08)))
    axes_1.set_ylim([-0.1, 1.1])
    # axes_1.set_autoscaley_on(False)

    axes_2 = fig.add_subplot(2, 3, 2)
    axes_2.plot(y_test[0, :, 1])
    axes_2.set_title('Left lane change label - Predicted')
    axes_2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '%g' % (x * 0.08)))
    axes_2.set_ylim([-0.1, 1.1])

    axes_3 = fig.add_subplot(2, 3, 3)
    axes_3.plot(y_test[0, :, 2])
    axes_3.set_title('Right lane change label - Predicted')
    axes_3.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '%g' % (x * 0.08)))
    axes_3.set_ylim([-0.1, 1.1])

    axes_4 = fig.add_subplot(2, 3, 4)
    axes_4.plot(Y[num, :, 0])
# <<<<<<< HEAD
#     axes_4.set_title('Free driving label - Real')
# =======
#     axes_4.set_title('Free driving label - Reference')
# >>>>>>> 8a8a09d237282c56f77286390fadbd3946ebb716
#     axes_4.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '%g' % (x * 0.08)))
#     axes_4.set_ylim([-0.1, 1.1])
#
#     axes_5 = fig.add_subplot(2, 3, 5)
#     axes_5.plot(Y[num, :, 1])
# <<<<<<< HEAD
#     axes_5.set_title('Left lane change label - Real')
# =======
#     axes_5.set_title('Left lane change label - Reference')
# >>>>>>> 8a8a09d237282c56f77286390fadbd3946ebb716
#     axes_5.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '%g' % (x * 0.08)))
#     axes_5.set_ylim([-0.1, 1.1])
#
#     axes_6 = fig.add_subplot(2, 3, 6)
#     axes_6.plot(Y[num, :, 2])
# <<<<<<< HEAD
#     axes_6.set_title('Right lane change label - Real')
#     axes_6.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '%g' % (x * 0.08)))
#     axes_6.set_ylim([-0.1, 1.1])
#
#     # fig.tight_layout()
# =======
#     axes_6.set_title('Right lane change label - Reference')
#     axes_6.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '%g' % (x * 0.08)))
#     axes_6.set_ylim([-0.1, 1.1])
#
# >>>>>>> 8a8a09d237282c56f77286390fadbd3946ebb716
#     plt.show()

# plt.subplot(2, 3, 1)
# plt.plot(y_test[0, :, 0])
# plt.title('Free driving label - Predicted ')
#
# plt.subplot(2, 3, 2)
# plt.plot(y_test[0, :, 1])
# plt.title('Left lane change label - Predicted ')
#
# plt.subplot(2, 3, 3)
# plt.plot(y_test[0, :, 2])
# plt.title('Right lane change label - Predicted ')
#
# plt.subplot(2, 3, 4)
# plt.plot(Y[num, :, 0])
# plt.title('Free driving label - Real')
#
# plt.subplot(2, 3, 5)
# plt.plot(Y[num, :, 1])
# plt.title('Left lane change label - Real')
#
# plt.subplot(2, 3, 6)
# plt.plot(Y[num, :, 2])
# plt.title('Right lane change label - Real')

# plt.set_xlim([-40, 40])
# plt.axis([None, None, -0.1, 1.1])
