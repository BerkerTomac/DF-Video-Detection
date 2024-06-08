import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from torch.nn.functional import relu, leaky_relu

class Model(nn.Module):
    def __init__(self, device):
        super(Model, self).__init__()
        self.device = device
        self.feature_extractor1 = efficientnet_b0(pretrained=True)
        self.feature_extractor2 = efficientnet_b0(pretrained=True)

        num_features = self.feature_extractor1.classifier[1].in_features
        self.feature_extractor1.classifier[1] = nn.Linear(num_features, 512)
        self.feature_extractor2.classifier[1] = nn.Linear(num_features, 512)

        self.lstm1 = nn.LSTM(1024, 256, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)  # Dropout layer with 50% dropout rate
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        video_frames, optical_flow_frames = x

        batch_size, seq_len, C, H, W = video_frames.size()

        video_frames = video_frames.view(batch_size * seq_len, C, H, W)
        optical_flow_frames = optical_flow_frames.view(batch_size * seq_len, C, H, W)

        # Extract features
        video_x = leaky_relu(self.feature_extractor1(video_frames))
        optical_flow_x = leaky_relu(self.feature_extractor2(optical_flow_frames))

        # Reshape back to (batch_size, seq_len, feature_size)
        video_x = video_x.view(batch_size, seq_len, -1)
        optical_flow_x = optical_flow_x.view(batch_size, seq_len, -1)

        x = torch.cat((video_x, optical_flow_x), dim=2)

        x, _ = self.lstm1(x)
        x = x[:, -1, :]  # Take the last output of the LSTM
        x = self.dropout(x)  # Apply dropout
        x = leaky_relu(x)
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = self.fc3(x)
        return x
