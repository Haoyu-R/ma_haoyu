import numpy as np

X = np.load(r'C:\Users\arhyr\Desktop\audi\ma_haoyu\processed_data\csv\test\X.npy')
Y = np.load(r'C:\Users\arhyr\Desktop\audi\ma_haoyu\processed_data\csv\test\Y.npy')
normalization = np.load(r'C:\Users\arhyr\Desktop\audi\ma_haoyu\processed_data\csv\test\mean_std.npy')
print(X[17, 0, 0])
print(X.shape)
print(Y.shape)
print(normalization.shape)
