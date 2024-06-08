import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Compose, Resize, ToTensor

class VideoDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        super(VideoDataset, self).__init__()
        self.dataset_path = dataset_path
        self.transform = transform
        self.frames_to_read = 10

        self.video_paths, self.labels = self._get_videos(self.dataset_path)

    def _get_videos(self, dataset_path: str):
        original_videos = list(os.listdir(os.path.join(dataset_path, 'original', 'faces')))
        manipulated_videos = list(os.listdir(os.path.join(dataset_path, 'manipulated', 'faces')))
        labels = [0] * len(original_videos) + [1] * len(manipulated_videos)
        videos = original_videos + manipulated_videos
        return videos, labels

    def get_random_start_frame(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if frame_count <= self.frames_to_read:
            return 0
        else:
            return random.randint(0, frame_count - self.frames_to_read)

    def read_video(self, video_path, start_frame):
        cap = cv2.VideoCapture(video_path)

        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(self.frames_to_read):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)

        cap.release()

        while len(frames) < self.frames_to_read:
            frames.append(torch.zeros_like(frames[0]))

        frames = torch.stack(frames)
        return frames

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        str_label = 'original' if label == 0 else 'manipulated'
        video_path = os.path.join(self.dataset_path, str_label, 'faces', self.video_paths[idx])
        optical_flow_video_path = video_path.replace('/', '\\').replace('\\faces\\', '\\optical\\')
        start_frame = self.get_random_start_frame(optical_flow_video_path)
        video_frames = self.read_video(video_path, start_frame)
        optical_flow_frames = self.read_video(optical_flow_video_path, start_frame)
        label = torch.tensor([label])
        return video_frames, optical_flow_frames, label
