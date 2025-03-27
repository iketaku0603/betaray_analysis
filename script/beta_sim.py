import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

# 4x4 の人工データを作成（1が信号、0が背景）
data = np.array([
    [0, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
])

# 閾値処理（今回はすでに0/1なので不要だけど明示的に）
binary_image = data > 0.5

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
