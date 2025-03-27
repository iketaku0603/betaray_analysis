import numpy as np
from scipy.ndimage import label

# ---------- クラスタリング関数 ----------

def extract_clusters_from_frame(frame_array, frame_id, threshold=1):
    """
    1枚のフレームに対してクラスタリングを行い、
    (frame_id, event_id, x, y, value) のリストを返す。
    """
    binary_image = frame_array >= threshold
    structure = np.ones((3, 3), dtype=int)
    labeled_array, num_clusters = label(binary_image, structure=structure)
    
    cluster_pixel_list = []
    
    for event_id in range(1, num_clusters + 1):
        indices = np.argwhere(labeled_array == event_id)
        for y, x in indices:
            value = frame_array[y, x]
            cluster_pixel_list.append((frame_id, event_id, x, y, value))
    
    return cluster_pixel_list

# ---------- テストフレーム作成（3枚） ----------

frames = [
    np.array([[1, 0, 0], [0, 2, 3], [0, 4, 0]]),
    np.array([[0, 0, 0], [5, 1, 0], [0, 0, 0]]),
    np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
]

# ---------- 全フレームに対して処理 ----------

all_cluster_data = []

for frame_id, frame_array in enumerate(frames):
    cluster_data = extract_clusters_from_frame(frame_array, frame_id, threshold=1)
    all_cluster_data.extend(cluster_data)

# ---------- 出力確認 ----------

for row in all_cluster_data:
    print(row)

#print(all_cluster_data)