import numpy as np
from scipy.ndimage import label
import matplotlib.pyplot as plt

# 実数値を持つ配列
test_array = np.array([
    [1, 0, 0],
    [0, 0, 3],
    [0, 4, 0]
])

# 閾値処理：1以上のピクセルをTrue、それ以外をFalseとする
threshold = 1
binary_image = test_array >= threshold

# クラスタリング（8近傍）
structure = np.ones((3, 3), dtype=int)
labeled_array, num_clusters = label(binary_image, structure=structure)

print("クラスタ数:", num_clusters)
print("ラベル付き配列:")
print(labeled_array)

# 可視化
plt.imshow(labeled_array, cmap='nipy_spectral', interpolation='nearest')
plt.title("Clustered Image")
plt.colorbar(label='Cluster ID')
plt.show()
