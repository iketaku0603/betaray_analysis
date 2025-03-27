import numpy as np
from scipy.ndimage import label
import os

# ---------- クラスタ抽出関数 ----------
def extract_clusters_from_frame(frame_array, frame_id, threshold=1):
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

# ---------- 連番CSVファイルを処理 ----------
data_folder = r"C:\Users\iketa\Experiment\test_beta\data\beta500000ns150V"

all_cluster_data = []

for i, event_num in enumerate(range(110, 200)):  # 110〜199の連番
    filename = os.path.join(data_folder, f"csvdata_event{event_num}.csv")
    frame_array = np.loadtxt(filename, delimiter=",")
    
    cluster_data = extract_clusters_from_frame(frame_array, i, threshold=1)
    all_cluster_data.extend(cluster_data)

# ---------- 出力確認 ----------
print(f"クラスタ数（行数）: {len(all_cluster_data)}")
for row in all_cluster_data[:5]:
    print(row)
