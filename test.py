import os
import pandas as pd
import numpy as np

a = np.random.random((5, 5))
b = np.random.random((5, 5))
c = np.random.random((5, 5))
d = np.concatenate((a, b, c), axis=0)
print(d)
print(d.shape)

