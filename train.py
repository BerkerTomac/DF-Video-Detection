import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose, ToTensor, Resize, Normalize
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from model import Model
from video_dataset import VideoDataset

import warnings

warnings.filterwarnings("ignore")


def train(model, device, train_dataloader, validation_dataloader, n_epoch=10):
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    file_save_path = r"D:\CMPE490\habunabibak"

    for epoch in range(n_epoch):
        print(f'Epoch {epoch + 1}/{n_epoch}')
        train_one_epoch(model, device, criterion, optimizer, train_dataloader)
        validate_model(model, validation_dataloader, criterion)
        if epoch == 0 or (epoch + 1) % 10 == 0:
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
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Train Loss (BCE): {running_loss / len(dataloader)}\n")


def custom_collate_fn(batch):
    video_frames, optical_flow_frames, labels = zip(*batch)

    # Stack the frames
    video_frames = torch.stack(video_frames)
    optical_flow_frames = torch.stack(optical_flow_frames)
    labels = torch.stack(labels)

    return video_frames, optical_flow_frames, labels


def validate_model(model, dataloader, criterion):
    print('Validating...')
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for video, optical_flow, label in tqdm(dataloader):
            video, optical_flow, label = video.to(device).float(), optical_flow.to(device).float(), label.to(
                device).float()

            output = model([video, optical_flow])
            loss = criterion(output, label)
            running_loss += loss.item()
            preds = (torch.sigmoid(output) > 0.5).float()
            running_accuracy += preds.eq(label).sum().item()

            all_labels.extend(label.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)

    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"Validation Loss (BCE): {avg_loss}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}\n")


device = 'cuda'
cont_path = r"D:\CMPE490\df_epoch_30.pth"
model = Model(device)

transform = Compose([
    ToTensor(),
    Resize((224, 224)),
    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = VideoDataset(r"D:\CMPE490-KOD\facefolder\train", transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=3, collate_fn=custom_collate_fn)

validation_dataset = VideoDataset(r"D:\CMPE490-KOD\facefolder\val", transform=transform)
validation_dataloader = DataLoader(validation_dataset, batch_size=8, shuffle=False, num_workers=3,
                                   collate_fn=custom_collate_fn)

if __name__ == '__main__':
    train(model, device, train_dataloader, validation_dataloader, n_epoch=100)
