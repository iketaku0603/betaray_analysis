import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

def visualize_clusters_in_red_filtered(filename, threshold=10, min_cluster_size=2):
    data = np.loadtxt(filename, delimiter=",")
    binary_image = data >= threshold
    structure = np.ones((3, 3), dtype=int)
    labeled_array, num_clusters = label(binary_image, structure=structure)

    print(f"全クラスタ数: {num_clusters}")

    # ピクセル数でふるいにかける
    filtered_mask = np.zeros_like(labeled_array, dtype=bool)
    kept_event_count = 0
    for cluster_id in range(1, num_clusters + 1):
        mask = (labeled_array == cluster_id)
        cluster_size = np.sum(mask)
        if cluster_size >= min_cluster_size:
            filtered_mask |= mask  # 条件を満たすクラスタだけ足す
            kept_event_count += 1

    print(f"{min_cluster_size}ピクセル以上のクラスタ数: {kept_event_count}")

    # 図示
    fig, ax = plt.subplots(figsize=(10, 8))

    heatmap = ax.imshow(data, cmap='viridis', interpolation='nearest')
    plt.colorbar(heatmap, ax=ax, label='Pixel Value')

    ys, xs = np.where(filtered_mask)
    ax.scatter(xs, ys, color='red', s=1)

    ax.set_title(f"Heatmap + Filtered Clusters (≥{min_cluster_size} px)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    plt.show()


filename = r"C:\Users\iketa\Experiment\betaray\betaray_analysis\data\beta500000ns150V\csvdata_event143.csv"
visualize_clusters_in_red_filtered(filename, threshold=20, min_cluster_size=4)
