import os

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

        self.video_paths, self.labels = self._get_videos(self.dataset_path)

    def _get_videos(self, dataset_path: str):
        original_videos = list(os.listdir(os.path.join(dataset_path, 'original')))
        manipulated_videos = list(os.listdir(os.path.join(dataset_path, 'manipulated')))
        labels = [0] * len(original_videos) + [1] * len(manipulated_videos)
        videos = original_videos + manipulated_videos
        return videos, labels

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        str_label = 'original' if label == 0 else 'manipulated'
        video_path = os.path.join(self.dataset_path, str_label, self.video_paths[idx])

        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)

        frames = torch.stack(frames)  # Convert list of frames to tensor
        label = torch.tensor(label)
        return frames, label