import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

file_name = r'history_4_GRU'

path = r'C:\Users\arhyr\Desktop\rec\{}.csv'
file = pd.read_csv(path.format(file_name))

fig = plt.figure(figsize=(12, 3))

axes_1 = fig.add_subplot(1, 2, 1)
axes_1.plot(file.loc[:, 'acc'], lw=1)
axes_1.plot(file.loc[:, 'val_acc'], lw=1)
axes_1.set_title('Model accuracy', fontsize=13)
# axes_1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '%g' % (x * 0.08)))
axes_1.set_ylim([0.92, 1])
# axes_1.set_ylim([0.85, 1])
axes_1.set_xlim([0, 500])
axes_1.set_xlabel('Epoch', fontsize=11)
axes_1.set_ylabel('Accuracy', fontsize=11)
axes_1.legend(['Train', 'Test'], loc='upper left', fontsize=10)
axes_1.grid('on')
axes_1.spines["top"].set_visible(False)
axes_1.spines["bottom"].set_visible(False)
axes_1.spines["right"].set_visible(False)
axes_1.spines["left"].set_visible(False)
axes_1.get_xaxis().tick_bottom()
axes_1.get_yaxis().tick_left()


axes_2 = fig.add_subplot(1, 2, 2)
axes_2.plot(file.loc[:, 'loss'], lw=1)
axes_2.plot(file.loc[:, 'val_loss'], lw=1)
axes_2.set_title('Model loss', fontsize=13)
# axes_2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '%g' % (x * 0.08)))
axes_2.set_xlim([0, 500])
axes_2.set_ylim([0, 0.2])
# axes_2.set_ylim([0, 0.7])
axes_2.set_xlabel('Epoch', fontsize=11)
axes_2.set_ylabel('Loss', fontsize=11)
axes_2.legend(['Train', 'Test'], loc='lower left', fontsize=10)
axes_2.grid('on')
axes_2.spines["top"].set_visible(False)
axes_2.spines["bottom"].set_visible(False)
axes_2.spines["right"].set_visible(False)
axes_2.spines["left"].set_visible(False)
axes_2.get_xaxis().tick_bottom()
axes_2.get_yaxis().tick_left()

plt.subplots_adjust(bottom=0.15, top=0.9, wspace=0.2)
plt.show()