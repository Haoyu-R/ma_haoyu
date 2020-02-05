import matplotlib.pyplot as plt
import numpy as np

x = [1.30, 1.46, 2.66, 1.56]
x = [2.66, 1.56, 1.46, 1.3]
plt.bar(np.arange(4), x)
plt.title('feature importance')
plt.xticks(np.arange(4), ['acc_x', 'steering_ang', 'acc_y', 'speed'])
plt.show()