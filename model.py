import torch
import torch.nn as nn
from torchvision.models import resnet50
from torch.nn.functional import relu, sigmoid
class Model(nn.Module):
    def __init__(self, sequence_length, device):
        super(Model, self).__init__()
        self.sequence_length = sequence_length
        self.device = device
        self.feature_extractor = resnet50(pretrained=True)

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.feature_extractor.fc = nn.Linear(2048, 512)
        self.lstm1 = nn.LSTM(512, 256)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x): # x = [batch, seq, ch, h, w]
        sequences = x.squeeze(0).split(self.sequence_length, dim=0) if x.shape[1] > self.sequence_length else x
        batch_size, seq_len, C, H, W = sequences[0].unsqueeze(0).size()
        h_n = torch.zeros(batch_size, seq_len, 256, device=self.device)
        c_n = torch.zeros(batch_size, seq_len, 256, device=self.device)
        for sequence in sequences:
            x = sequence
            x = x.unsqueeze(0)
            batch_size, seq_len, C, H, W = x.size()
            # Flatten the temporal dimension and batch dimension together
            x = x.view(batch_size * seq_len, C, H, W)
            # Extract features
            x = self.feature_extractor(x)
            # Adjust dimensions
            x = x.view(batch_size, seq_len, -1)

            h_n = h_n[:, -seq_len:, :]
            c_n = c_n[:, -seq_len:, :]

            # LSTM processing
            lstm_out, (h_n, c_n) = self.lstm1(x, (h_n, c_n))
        # We use the last hidden state
        lstm_out = lstm_out[:, -1, :]

        # Pass through the fully connected layer
        out = relu(self.fc1(lstm_out))
        out = relu(self.fc2(out))
        out = sigmoid(self.fc3(out))
        return out

    '''
        def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        # Flatten the temporal dimension and batch dimension together
        x = x.view(batch_size * timesteps, C, H, W)
        # Extract features
        x = self.feature_extractor(x)
        # Adjust dimensions
        x = x.view(batch_size, timesteps, -1)

        # LSTM processing
        lstm_out, _ = self.lstm1(x)
        # We use the last hidden state
        lstm_out = lstm_out[:, -1, :]

        # Pass through the fully connected layer
        out = relu(self.fc1(lstm_out))
        out = relu(self.fc2(out))
        out = self.fc3(out)
        return out
    '''