import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchmetrics.classification import MultilabelAccuracy
import os
import pandas as pd

from models.LeNet import LeNet
from models.AlexNet import AlexNet
from models.ZFNet import ZFNet
from datasets.MiasDS import MiasDS

# Loading metadata
cwd = os.getcwd()
df = pd.read_csv(os.path.join(cwd, 'csv/new_mias.csv'))
db_path = '/Volumes/Seagate/monografia/all-mias/roi'

# Filter only the train images
df = df[df['db'] == 'train']
print('Total Filtered:', df['ref'].count())

# Check data
print(df.head())
print(df[
    (df['severity'] == 'N')
]['ref'].count())
print(df[
    (df['severity'] == 'M')
]['ref'].count())
print(df[
    (df['severity'] == 'B')
]['ref'].count())

# Generate the dummies for the severity column: N, M, B
df_dummies = pd.get_dummies(df['severity'], prefix='sev')
df = df_dummies.join(df['ref'])
print(df.head())

# Hyperparameters
LR = 0.001 #Learning rate
BATCH_SIZE = 64
EPOCHS = 50

# Pytorch device to use. In this case using 
# Metal Performance Shaders implementation for Mac devices
device = torch.device('mps')

# Train/Validation splits
train_split = 0.8
valid_split = 0.2

# Using the custom Torch dataset 
ds = MiasDS(df, db_path)
(ds_train, ds_valid) = random_split(ds, [train_split, valid_split], generator=torch.Generator().manual_seed(42))

# Creating data loaders 
train_data = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
valid_data = DataLoader(ds_valid, batch_size=BATCH_SIZE)

# Instantiate the custom CNN model
model = LeNet()
model.to(device)

# Loss function and Optmizer initialization
loss_func = nn.CrossEntropyLoss()
opt = torch.optim.Adam(params=model.parameters(), lr=LR)

print('Start training....')

## Training model
metric = MultilabelAccuracy(3, threshold=0.5)

# Iterate over defined Epochs
for epoch in range(0, EPOCHS):
    model.train()

    # Iterate over training dataset
    for (images, targets) in train_data:
        # Send tensors to the device
        images = images.to(device)
        targets = targets.to(device, dtype=torch.float32)

        # Forward propagation
        predicts = model(images)
        loss = loss_func(predicts, targets)

        # Backward propagation and optimization
        opt.zero_grad()
        loss.backward()
        opt.step()

    preds = torch.tensor([])
    targs = torch.tensor([])
    with torch.no_grad():
        model.eval()

        # Iterate over validation dataset
        for (images, targets) in valid_data:
            # Send tensors to the device
            images = images.to(device)
            targets = targets.to(device, dtype=torch.float32)

            # Predict classes
            predicts = model(images)
            loss = loss_func(predicts, targets)

            # Calculate accuracy
            preds = torch.cat((preds, predicts.to('cpu')), dim=0)
            targs = torch.cat((targs, targets.to('cpu')), dim=0)

    print('Epoch: {} | Loss: {:.3f} | Accuracy: {:.2%}'.format(epoch + 1, loss.item(), metric(preds, targs)))

torch.save(model.state_dict(), os.path.join(cwd, 'model/model.torch'))
