import sys
import os
import numpy as np
import pickle
from PIL import Image

def cross_entropy_errror(y, t):
    delta = 1e-7    # オーバーフロー対策
    return -np.sum(t * np.log(y + delta))

# 教師データ
t = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
# 出力
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

loss = cross_entropy_errror(np.array(y), np.array(t))

print(loss)