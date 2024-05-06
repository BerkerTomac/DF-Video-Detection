import math

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose, ToTensor, Resize
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

from model import Model
from video_dataset import VideoDataset

# Import torch.cuda.amp for automatic mixed precision (AMP)
from torch.cuda.amp import autocast, GradScaler


def train(model, device, train_dataloader, validation_dataloader, n_epoch=10):
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Initialize GradScaler for automatic mixed precision
    scaler = GradScaler()

    for epoch in range(n_epoch):
        print(f'Epoch {epoch+1}/{n_epoch}')
        train_one_epoch(model, device, criterion, optimizer, train_dataloader, scaler)
        validate_model(model, validation_dataloader, criterion)
        if epoch == 0 or (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'D:\\CMPE490df_epoch_{epoch + 1}.pth')


def train_one_epoch(model, device, criterion, optimizer, dataloader, scaler):
    print('Training...')
    model.train()
    running_loss = 0

    for video, label in tqdm(dataloader):
        video, label = video.to(device).float(), label.to(device).float()

        # Wrap the training loop with autocast to enable mixed precision
        with autocast():
            optimizer.zero_grad()
            output = model(video)
            output = output.squeeze(0)
            loss = criterion(output, label)

        # Perform backpropagation using GradScaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    print(f"Train Loss (BCE): {running_loss / len(dataloader)}\n")


def validate_model(model, dataloader, criterion):
    print('Validating...')
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0

    with torch.no_grad():
        for video, label in tqdm(dataloader):
            video, label = video.to(device).float(), label.to(device).float()

            output = model(video)
            output = output.squeeze(0)
            loss = criterion(output, label)
            running_loss += loss.item()
            running_accuracy += (torch.where(output >= 0.5, 1, 0) == label).sum().to('cpu').numpy()

    avg_loss = running_loss / len(dataloader)
    accuracy = running_accuracy / len(dataloader.dataset)
    print(f"Validation Loss (BCE): {avg_loss}")
    print(f"Accuracy: {accuracy}\n")


device = 'cuda'
model = Model(600, device)

transforms = Compose([
        ToTensor(),
        Resize((224, 224))
    ])
train_dataset = VideoDataset(r"D:\CMPE490-KOD\dataset\train", transform=transforms)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle= True, num_workers= 2)

validation_dataset = VideoDataset(r"D:\CMPE490-KOD\dataset\val", transform=transforms)
validation_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=2)

if __name__ == '__main__':
    train(model, device, train_dataloader, validation_dataloader, n_epoch=10)
