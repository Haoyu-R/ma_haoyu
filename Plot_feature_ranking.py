import matplotlib.pyplot as plt
import numpy as np

# x = [2.12, 1.80, 1.76, 1.68, 1.59, 1.38, 1.37, 1.21, 1.15]
# x = [i/2.12 for i in x]
# rows = ['pos_y_obj1', 'acc_x', 'speed_x_obj1', 'speed_y_obj1', 'acc_y', 'pos_x_obj1', 'speed', 'pos_x_obj2', 'steering_ang']

x = [3.22, 1.65, 1.49, 1.30, 1.29, 1.23, 1.14, 1.14, 1.10]
x = [i/3.22 for i in x]
rows = ['acc_y', 'acc_x', 'speed', 'pos_y_obj1', 'speed_x_obj1', 'speed_y_obj1', 'steering_ang', 'pos_x_obj1', 'pos_y_obj2']



x.reverse()
rows.reverse()
plt.title('Normalized relative feature importance of lane-change scenario')
plt.barh(np.arange(9), x, align='center', linewidth=0, alpha=0.7)
plt.yticks(np.arange(9), rows)
plt.subplots_adjust(bottom=0.15, top=0.7, wspace=0.2, left=0.2, right=0.95)
plt.show()