import numpy as np
from scipy.ndimage import label

# 3x4 の単純配列
test_arr = np.array([
    [0,1,1,0],
    [1,1,0,0],
    [0,0,0,1]
], dtype=int)

# 5x5 の構造体
structure_5x5 = np.ones((3,3), dtype=int)

print("test_arr shape =", test_arr.shape)
print("structure shape =", structure_5x5.shape)

labeled, n = label(test_arr, structure=structure_5x5)
print("labeled =", labeled)
print("num_clusters =", n)
