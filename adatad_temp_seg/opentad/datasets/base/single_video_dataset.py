import os
import numpy as np
from mmengine.dataset import BaseDataset
from mmengine.dataset import Compose
from mmengine.registry import DATASETS

from ..builder import DATASETS, get_class_index
import decord

@DATASETS.register_module()
class SingleVideoDataset(BaseDataset):
    def __init__(
        self,
        video_path,
        pipeline,
        class_map,
        feature_stride=4,
        sample_stride=1,
        window_size=768,
        window_overlap_ratio=0.5,
        test_mode=True,
        logger=None,
        **kwargs
    ):
        super().__init__(pipeline=pipeline, test_mode=test_mode, lazy_init=True, **kwargs)
        self.video_path = video_path
        self.class_map = self.get_class_map(class_map)
        self.feature_stride = feature_stride
        self.sample_stride = sample_stride
        self.snippet_stride = self.feature_stride * self.sample_stride
        self.window_size = window_size
        self.window_overlap_ratio = window_overlap_ratio
        self.test_mode = test_mode
        self.logger = logger.info if logger is not None else print

        # Extract video information
        self.video_info = self.get_video_info()
        # Split the video into windows
        self.data_list = self.split_video_to_windows()

        # Set _fully_initialized to True to prevent BaseDataset from re-initializing
        self._fully_initialized = True

    def get_class_map(self, class_map_path):
        with open(class_map_path, 'r') as f:
            lines = f.readlines()
        class_map = [line.strip() for line in lines]
        return class_map

    def get_video_info(self):
        vr = decord.VideoReader(self.video_path)
        num_frames = len(vr)
        fps = vr.get_avg_fps()
        duration = num_frames / fps
        video_info = {
            'frame': num_frames,
            'duration': duration,
            'fps': fps,
            'video_path': self.video_path
        }
        return video_info

    def split_video_to_windows(self):
        num_frames = self.video_info['frame']
        snippet_stride = self.snippet_stride
        window_size = self.window_size
        window_stride = int(window_size * (1 - self.window_overlap_ratio))
        video_snippet_centers = np.arange(0, num_frames, snippet_stride)
        snippet_num = len(video_snippet_centers)

        data_list = []
        last_window = False
        
        for idx in range(max(1, snippet_num // window_stride)):
            window_start = idx * window_stride
            window_end = window_start + self.window_size

            if window_end > snippet_num:
                window_end = snippet_num
                window_start = max(0, window_end - self.window_size)
                last_window = True

            window_snippet_centers = video_snippet_centers[window_start:window_end]

            data = dict(
                video_path=self.video_path,
                window_snippet_centers=window_snippet_centers,
                video_info=self.video_info,
                feature_start_idx=window_start,
                feature_end_idx=window_end - 1,  # Added line
            )
            data_list.append(data)

            if last_window:
                break

        return data_list


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        results = dict()
        results['data_path'] = os.path.dirname(self.video_path)
        results['video_path'] = data['video_path']
        results['video_name'] = os.path.splitext(os.path.basename(self.video_path))[0]
        results['duration'] = self.video_info['duration']
        results['fps'] = self.video_info['fps']
        results['total_frames'] = self.video_info['frame']
        results['snippet_centers'] = data['window_snippet_centers']
        results['snippet_stride'] = self.snippet_stride
        results['window_size'] = self.window_size
        results['feature_start_idx'] = data['feature_start_idx']
        results['feature_end_idx'] = data['feature_end_idx']
        # Calculate and add 'offset_frames'
        results['offset_frames'] = results['feature_start_idx'] * results['snippet_stride']
        results['video_info'] = data['video_info']
        results = self.pipeline(results)
        return results
    
    def load_data_list(self):
        """Load the data list.

        Overrides the BaseDataset method to return the data list that
        was prepared in the __init__ method.
        """
        return self.data_list
