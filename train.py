import math
import os

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.v2 import Compose, ToTensor, Resize
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

from model import Model
from video_dataset import VideoDataset

'''device = 'cuda'
model = Model()
model.to(device)
lstm_out_dim = 1280

h_0 = torch.zeros(1, 1, 500, device=device)
c_0 = torch.zeros(1, 1, 500, device=device)

transforms = Compose([
        ToTensor(),
        Resize((224, 224))
    ])
dataset = VideoDataset(r'D:\Dosyalarim\Belgelerim\Bitirme\VenatorAnalyzer\berker\dataset\train', transform=transforms)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
for video, label in dataloader:
    video, label = video.to(device), label.to(device)
    for frame in video.squeeze(0):
        output, (h_n, c_n) = model(frame.unsqueeze(0), h_0=h_0, c_0=c_0)
        print(output.shape)
        print(h_n.shape)
        print(c_n.shape)
        break
    break'''


def train(model, device, train_dataloader, validation_dataloader, n_epoch=10):
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Model file save path
    file_save_path = r"D:\CMPE490"

    torch.save(model.state_dict(), os.path.join(file_save_path, 'df_save_test.pth'))
    for epoch in range(n_epoch):
        print(f'Epoch {epoch+1}/{n_epoch}')
        train_one_epoch(model, device, criterion, optimizer, train_dataloader)
        validate_model(model, validation_dataloader, criterion)
        if epoch == 0 or (epoch + 1) % 5 == 0:
            try:
                torch.save(model.state_dict(), os.path.join(file_save_path, f'df_epoch_{epoch + 1}.pth'))
            except Exception as e:
                print('Model file cannot be saved.\n' + str(e))


def train_one_epoch(model, device, criterion, optimizer, dataloader):
    print('Training...')
    model.train()
    running_loss = 0
    for video, optical_flow, label in tqdm(dataloader):
        video, optical_flow, label = video.to(device).float(), optical_flow.to(device).float(), label.to(device).float()

        optimizer.zero_grad()
        output = model([video, optical_flow])
        output = output.squeeze(0)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Train Loss (BCE): {running_loss / len(dataloader)}\n")

def validate_model(model, dataloader, criterion):
    print('Validating...')
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    with torch.no_grad():
        for video, optical_flow, label in tqdm(dataloader):
            video, optical_flow, label = video.to(device).float(), optical_flow.to(device).float(), label.to(device).float()

            output = model([video, optical_flow])
            output = output.squeeze(0)
            loss = criterion(output, label)
            running_loss += loss.item()
            running_accuracy += (torch.where(output >= 0.5, 1, 0) == label).sum().to('cpu').numpy()

    avg_loss = running_loss / len(dataloader)
    accuracy = running_accuracy / len(dataloader.dataset)
    print(f"Validation Loss (BCE): {avg_loss}")
    print(f"Accuracy: {accuracy}\n")


device = 'cuda'
model = Model(200, device)

train_transforms = Compose([
    ToTensor(),
    transforms.v2.RandomHorizontalFlip(p=0.5),
    transforms.v2.RandomVerticalFlip(p=0.1),
    transforms.v2.RandomRotation((-45, 45)),
    transforms.v2.RandomPerspective(distortion_scale=0.4, p=0.15),
    transforms.v2.RandomGrayscale(p=0.1),
    transforms.v2.RandomApply(torch.nn.ModuleList([transforms.v2.ColorJitter()]), p=0.2),
    transforms.v2.RandomApply(torch.nn.ModuleList([transforms.v2.GaussianBlur(kernel_size=5)]), p=0.2),
    Resize((224, 224))
    ])

val_transforms = Compose([
    ToTensor(),
    Resize((224, 224))
    ])
train_dataset = VideoDataset(r"D:\CMPE490-KOD\facefolder\train", transform=train_transforms)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=3)

validation_dataset = VideoDataset(r"D:\CMPE490-KOD\facefolder\val", transform=val_transforms)
validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=3)

if __name__ == '__main__':
    train(model, device, train_dataloader, validation_dataloader, n_epoch=20)