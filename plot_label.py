import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

a = np.ones(500)
b = np.zeros(500)
c = np.zeros(500)
d = np.zeros(500)
e = np.zeros(500)

a[50:74] = 0
a[300:324] = 0
a[450:474] = 0

b[50:74] = 1
d[300:324] = 1
c[450:474] = 1

fig = plt.figure()

axes_1 = fig.add_subplot(5, 1, 1)
axes_1.plot(a[:])
axes_1.set_title('Free driving label')
axes_1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '%g' % (x * 0.04)))
axes_1.set_ylim([-0.1, 1.1])

axes_2 = fig.add_subplot(5, 1, 2)
axes_2.plot(b[:])
axes_2.set_title('Left lane change label')
axes_2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '%g' % (x * 0.04)))
axes_2.set_ylim([-0.1, 1.1])


axes_3 = fig.add_subplot(5, 1, 3)
axes_3.plot(c[:])
axes_3.set_title('Right lane change label')
axes_3.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '%g' % (x * 0.04)))
axes_3.set_ylim([-0.1, 1.1])

axes_4 = fig.add_subplot(5, 1, 4)
axes_4.plot(d[:])
axes_4.set_title('Cut-in from left label')
axes_4.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '%g' % (x * 0.04)))
axes_4.set_ylim([-0.1, 1.1])

axes_5 = fig.add_subplot(5, 1, 5)
axes_5.plot(e[:])
axes_5.set_title('Cut-in from right label')
axes_5.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '%g' % (x * 0.04)))
axes_5.set_ylim([-0.1, 1.1])
plt.show()



