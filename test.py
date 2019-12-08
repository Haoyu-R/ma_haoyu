import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import math

for i in range(600):
    if i > 300:
        print(float(0.001 * math.exp(0.1 * int(300 - i))))

