import sys
import os
import numpy as np
import pickle
from PIL import Image

def sum_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

# 教師データ
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
# 出力
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

loss = sum_squared_error(np.array(y), np.array(t))

print(loss)