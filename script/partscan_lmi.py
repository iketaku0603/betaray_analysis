import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ファイルパスのベースディレクトリ
base_path = r'C:\Users\iketa\Experiment\betaray\beta500000ns150V'

# 平均値と標準偏差のCSVファイルの読み込み
data_avg = pd.read_csv(fr'{base_path}\csvdata_avg.csv', header=None)
avg_event_values = data_avg.values.flatten()
avg_event_mean = avg_event_values.mean()
avg_event_stddev = avg_event_values.std()
threshold = avg_event_mean + 3 * avg_event_stddev
threshold = 10

# 解析範囲（全範囲）
x_start, x_end = 0, data_avg.shape[1]
y_start, y_end = 0, data_avg.shape[0]

# フレーム番号（今回は仮に全て0とする）
frame_number = 0

# イベントの範囲
start = 142
end = 142

# 結果を格納するリスト
list_data = []


def run_clustering(data, threshold):
    clusters = {}
    current_cluster_id = 0

    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            if data[x, y] > threshold:
                cluster_found = False

                # 上
                if x > 0 and data[x - 1, y] > threshold:
                    clusters[(x, y)] = clusters.get((x - 1, y), current_cluster_id)
                    cluster_found = True
                    if y > 0 and data[x, y - 1] > threshold:
                        left_cluster_id = clusters.get((x, y - 1))
                        if left_cluster_id is not None and left_cluster_id != clusters[(x, y)]:
                            for key, value in clusters.items():
                                if value == left_cluster_id:
                                    clusters[key] = clusters[(x, y)]

                # 左
                if y > 0 and data[x, y - 1] > threshold and not cluster_found:
                    clusters[(x, y)] = clusters.get((x, y - 1), current_cluster_id)
                    cluster_found = True
                    if x > 0 and y < data.shape[1] - 1 and data[x - 1, y + 1] > threshold:
                        top_right_cluster_id = clusters.get((x - 1, y + 1))
                        if top_right_cluster_id is not None and top_right_cluster_id != clusters[(x, y)]:
                            for key, value in clusters.items():
                                if value == top_right_cluster_id:
                                    clusters[key] = clusters[(x, y)]

                # 左上
                if x > 0 and y > 0 and data[x - 1, y - 1] > threshold and not cluster_found:
                    clusters[(x, y)] = clusters.get((x - 1, y - 1), current_cluster_id)
                    cluster_found = True
                    if y < data.shape[1] - 1 and data[x - 1, y + 1] > threshold:
                        top_right_cluster_id = clusters.get((x - 1, y + 1))
                        if top_right_cluster_id is not None and top_right_cluster_id != clusters[(x, y)]:
                            for key, value in clusters.items():
                                if value == top_right_cluster_id:
                                    clusters[key] = clusters[(x - 1, y - 1)]

                # 右上
                if x > 0 and y < data.shape[1] - 1 and data[x - 1, y + 1] > threshold and not cluster_found:
                    clusters[(x, y)] = clusters.get((x - 1, y + 1), current_cluster_id)
                    cluster_found = True

                # 新規クラスター作成
                if not cluster_found:
                    clusters[(x, y)] = current_cluster_id
                    current_cluster_id += 1

    return clusters



# 各イベントに対する処理
for event_number in range(start, end + 1):
    print(f"\n--- イベント {event_number} の処理を開始 ---")
    
    # データ読み込み
    filepath = fr'{base_path}\csvdata_event{event_number}.csv'
    data_event = pd.read_csv(filepath, header=None)
    data = data_event.iloc[y_start:y_end, x_start:x_end].values

    # クラスタリング実行
    clusters = run_clustering(data, threshold)

    # クラスタサイズを計算
    cluster_sizes = {}
    for cluster_id in clusters.values():
        cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1

    # サイズが5以上のクラスターを抽出
    filtered_clusters = {pos: cid for pos, cid in clusters.items() if cluster_sizes[cid] >= 5}

    # クラスターごとにピクセル情報をまとめる
    cluster_pixel_map = {}
    for (x, y), cid in filtered_clusters.items():
        val = data[x, y]
        cluster_pixel_map.setdefault(cid, []).append((y + x_start, x + y_start, val))  # (x, y, value)


    # # list_data にまとめる
    
    

    # for cid, pixels in cluster_pixel_map.items():
    #     total_value = sum(val for _, _, val in pixels)
    #     area = len(pixels)
    #     list_data.append([frame_number, event_number, total_value, area, pixels])

    #     # 表示（必要に応じて削除OK）
    #     print(f"イベント{event_number}, クラスター{cid}, 面積: {area}, 合計値: {total_value:.2f}")

    # 最大値による追加のふるいがけ
    max_value_threshold = 100  # ← ここで最大値のしきい値を定義（任意）

    for cid, pixels in cluster_pixel_map.items():
        total_value = sum(val for _, _, val in pixels)
        area = len(pixels)
        max_value = max(val for _, _, val in pixels)

        if max_value >= max_value_threshold:
            list_data.append([frame_number, event_number, total_value, area, pixels])
            print(f"イベント{event_number}, クラスター{cid}, 面積: {area}, 合計値: {total_value:.2f}, 最大値: {max_value:.1f}")
        else:
            print(f"イベント{event_number}, クラスター{cid} は最大値 {max_value:.1f} < {max_value_threshold} のため除外")


def plot_frame_with_clusters(frame_number_to_plot, list_data, base_path):
    # フレームに含まれるイベント番号を抽出
    event_numbers = set([row[1] for row in list_data if row[0] == frame_number_to_plot])
    
    if not event_numbers:
        print(f"フレーム番号 {frame_number_to_plot} に該当するイベントがありません。")
        return

    for event_number in event_numbers:
        # データの読み込み
        filepath = fr'{base_path}\csvdata_event{event_number}.csv'
        data = pd.read_csv(filepath, header=None).values

        # クラスターピクセルの抽出（クラスターごとに整理）
        clusters_in_frame = [
            row for row in list_data
            if row[0] == frame_number_to_plot and row[1] == event_number
        ]

        # プロット開始
        plt.figure(figsize=(8, 6))
        plt.imshow(data, cmap='gray', interpolation='nearest')
        plt.colorbar(label='Pixel Value')
        plt.title(f'Heatmap for Frame {frame_number_to_plot}, Event {event_number}')
        plt.xlabel('X')
        plt.ylabel('Y')

        # 各クラスターごとに青い点と赤い星をプロット
        for cluster in clusters_in_frame:
            _, _, _, _, pixels = cluster

            # 青い点でクラスターピクセル全体をプロット
            xs = [px[0] for px in pixels]
            ys = [px[1] for px in pixels]
            plt.scatter(xs, ys, color='blue', s=10)

            # 最大値ピクセルを取得して赤い星でマーク
            max_pixel = max(pixels, key=lambda p: p[2])  # p = (x, y, value)
            plt.scatter(max_pixel[0], max_pixel[1], color='red', marker='*', s=100)

        plt.tight_layout()
        plt.show()

frame_number_to_plot = 0
plot_frame_with_clusters(frame_number_to_plot, list_data, base_path)

