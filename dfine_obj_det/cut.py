import cv2
import os

def extract_clips(video_path, segments, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    for idx, segment in enumerate(segments):
        start, end = segment['segment']
        start_frame = int(start * fps)
        end_frame = int(end * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        output_path = os.path.join(output_dir, f"{prefix}_clip_{idx + 1}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for frame_num in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        out.release()
    cap.release()

bridge_segments = [{'segment': [0.0, 2.71], 'score': 0.0},
{'segment': [2.71, 7.39], 'score': 0.0},
{'segment': [7.39, 10.62], 'score': 0.0},
{'segment': [10.62, 12.78], 'score': 0.0},
 {'segment': [12.78, 17.52], 'score': 0.0},
{'segment': [17.52, 20.83], 'score': 0.0},
{'segment': [20.83, 24.98], 'score': 0.0},
{'segment': [24.98, 27.57], 'score': 0.0},
{'segment': [27.57, 30.68], 'score': 0.0},
{'segment': [30.68, 33.77], 'score': 0.0},
{'segment': [33.77, 36.97], 'score': 0.0},
{'segment': [36.97, 43.05], 'score': 0.0},
{'segment': [43.05, 46.22], 'score': 0.0},
{'segment': [46.22, 49.58], 'score': 0.0},
{'segment': [49.58, 54.05], 'score': 0.0},
{'segment': [54.05, 58.58], 'score': 0.0}]

video_path = "/data/enze/test_data/test_cases/quiet_1.mp4"
output_dir = "quiet_clips"
extract_clips(video_path, bridge_segments, output_dir, "quiet_1")