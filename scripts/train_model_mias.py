
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
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
df.head()

print(df.head())
print(df[
    (df['severity'] == 1)
].count())
print(df[
    (df['severity'] == 0)
].count())

# Hyperparameters
LR = 0.001 #Learning rate
BATCH_SIZE = 64
EPOCHS = 20

# Train/Validation splits
train_split = 0.6
valid_split = 0.2
test_split = 0.2

# Pytorch device to use. In this case using 
# Metal Performance Shaders implementation for Mac devices
device = torch.device('mps')

# Using the custom Torch dataset 
ds = MiasDS(df, db_path)

(ds_train, ds_valid, ds_test) = random_split(ds, [train_split, valid_split, test_split], generator=torch.Generator().manual_seed(42))

# Creating data loaders 
train_data = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
valid_data = DataLoader(ds_valid, batch_size=BATCH_SIZE)
test_data = []

# Instantiate the custom CNN model
model = LeNet()
model.to(device)

# Loss function and Optmizer initialization
loss_func = nn.BCELoss()
opt = torch.optim.Adam(params=model.parameters(), lr=LR)

print('Start training....')

## Training model
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

    correct_class = 0
    correct_severity = 0
    total = 0
    
    with torch.no_grad():
        model.eval()

        # Iterate over validation dataset
        for (images, targets) in valid_data:
            # Send tensors to the device
            images = images.to(device)
            targets = targets.to(device, dtype=torch.float32)

            # Forward propagation
            predicts = model(images)
            loss = loss_func(predicts, targets)

            total += len(targets)
            correct_class += [int(round(predicts[i][0].item(), 0)) == targets[i][0].item() for i in range(0, len(predicts))].count(True)
            correct_severity += [int(round(predicts[i][1].item(), 0)) == targets[i][1].item() for i in range(0, len(predicts))].count(True)

    print('Epoch: {} | Loss: {:.3f} | Accuracy Class: {:.2f}% | Accuracy Severity: {:.2f}%'.format(epoch + 1, loss.item(), correct_class / total * 100, correct_severity / total * 100))

torch.save(model.state_dict(), os.path.join(cwd, 'model/model.torch'))
