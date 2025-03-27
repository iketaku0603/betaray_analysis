import numpy as np
import os
from skimage.measure import label

def cluster_frames_to_nested_structure(folder_path, start_num, end_num, threshold, min_cluster_size):
    frames = []

    for i, event_num in enumerate(range(start_num, end_num + 1)):
        filepath = os.path.join(folder_path, f"csvdata_event{event_num}.csv")
        frame_array = np.loadtxt(filepath, delimiter=",")
        binary_image = frame_array >= threshold
        labeled = label(binary_image, connectivity=2)
        num_clusters = labeled.max()

        frame_info = {
            "frame_id": i,
            "num_events": 0,
            "events": []
        }

        # for event_id in range(1, num_clusters + 1):
        #     mask = (labeled == event_id)
        #     if np.sum(mask) >= min_cluster_size:
        kept_event_id = 1  # 条件を満たすイベントだけ連番にす
        for raw_id in range(1, num_clusters + 1):
             mask = (labeled == raw_id)
             if np.sum(mask) >= min_cluster_size:
                
                ys, xs = np.where(mask)
                pixels = []
                pixel_sum = 0.0
                for y, x in zip(ys, xs):
                    val = frame_array[y, x]
                    pixels.append({"x": int(x), "y": int(y), "value": float(val)})
                    pixel_sum += val

                frame_info["events"].append({
                    "event_id": kept_event_id,
                    "pixel_sum": float(pixel_sum),
                    "pixels": pixels
                })
                kept_event_id += 1
        frame_info["num_events"] = len(frame_info["events"])
        frames.append(frame_info)
    
    return frames

if __name__ == "__main__":
    folder = r"C:\Users\iketa\Experiment\betaray\betaray_analysis\data\beta500000ns150V"
    threshold = 10
    min_cluster_size = 5
    start_event = 142
    end_event = 142

    nested_cluster_data = cluster_frames_to_nested_structure(
        folder_path=folder,
        start_num=start_event,
        end_num=end_event,
        threshold=threshold,
        min_cluster_size=min_cluster_size
    )

    print(f"フレーム数: {len(nested_cluster_data)}")
    for frame in nested_cluster_data:
        print(f"Frame {frame['frame_id']} - {frame['num_events']} clusters")
        for event in frame['events']:
            print(f"  Event {event['event_id']}: sum = {event['pixel_sum']:.2f}, pixels = {len(event['pixels'])}")
