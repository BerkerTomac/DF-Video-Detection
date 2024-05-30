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
        self.lstm1 = nn.LSTM(1024, 256, batch_first=True)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

    def __feed_to_cnn(self, model, x):
        x = x.unsqueeze(0)
        batch_size, seq_len, C, H, W = x.size()
        # Flatten the temporal dimension and batch dimension together
        x = x.view(batch_size * seq_len, C, H, W)
        # Extract features
        x = self.feature_extractor(x)
        # Adjust dimensions
        x = x.view(batch_size, seq_len, -1)
        return x

    def forward(self, x): # x = [batch, seq, ch, h, w]
        video_frames = x[0]
        optical_flow_frames = x[1]
        #if video_frames.shape != optical_flow_frames.shape:
        #    raise ValueError("Number of optical flow frames must be equal to number of video frames.")
        video_shape = list(video_frames.shape)  # [100, 3, 224, 224] -> [1, 100, 3, 224, 224]
        video_shape[1] = 1
        blank_frame = torch.zeros(tuple(video_shape), device=self.device)  # (batch_size, 1, 3, 224, 224)
        optical_flow_frames = torch.cat((blank_frame, optical_flow_frames), dim=1)

        video_sequences = video_frames.squeeze(0).split(self.sequence_length, dim=0) #if video_frames.shape[1] > self.sequence_length else video_frames
        of_sequences = optical_flow_frames.squeeze(0).split(self.sequence_length, dim=0) #if optical_flow_frames.shape[1] > self.sequence_length else optical_flow_frames

        batch_size, seq_len, C, H, W = video_sequences[0].unsqueeze(0).size()
        h_n = torch.zeros(1, batch_size, 256, device=self.device)
        c_n = torch.zeros(1, batch_size, 256, device=self.device)

        for i in range(len(video_sequences)):
            video_x = video_sequences[i]
            of_x = of_sequences[i]
            video_x = self.__feed_to_cnn(self.feature_extractor, video_x)
            of_x = self.__feed_to_cnn(self.feature_extractor, of_x)

            x = torch.cat((video_x, of_x), dim=2)

            #hidden_state_size = x.shape[1] #min(seq_len, x.shape[1])
            #h_n = h_n[:, -hidden_state_size:, :]
            #c_n = c_n[:, -hidden_state_size:, :]

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