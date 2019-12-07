import numpy as np
import random

a = np.array([[10, 11, 12],
       [13, 14, 15]])
np.random.permutation(a)
print(a)
print(np.random.permutation(a))

# b = a[1,:]
# random.shuffle(b)
# a[1,:] = b