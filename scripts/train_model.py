import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import pandas as pd

from models.LeNet import LeNet
from models.AlexNet import AlexNet
from models.ZFNet import ZFNet
from datasets.CustomDS import CustomDS

# Loading metadata
cwd = os.getcwd()
df = pd.read_csv(os.path.join(cwd, 'csv/metadb_train.csv'))
db_path = '/Volumes/Seagate/monografia/database/roi/train/'

## Preping metadata
# Setting 1 to Malignant and 0 to Benign
df['pathology'] = df['pathology'].apply(lambda p: 1 if p == 'MALIGNANT' else 0)
# Setting 1 to Calc and 0 to Mass
df['type'] = df['type'].apply(lambda p: 1 if p == 'Calc' else 0)

#Maximum size allowed for the ROI images
IMG_SIZE = (700, 700)

print(df.head())
print('Total:', df['id'].count())
print('Bigger than {}x{}:'.format(IMG_SIZE[0], IMG_SIZE[1]), df[
    (df['width'] > IMG_SIZE[0]) |
    (df['height'] > IMG_SIZE[1])
]['id'].count())

#Filter images that are IMG_SIZE or less
df = df[
    (df['width'] <= IMG_SIZE[0]) &
    (df['height'] <= IMG_SIZE[1])
]
print('Total filtered:', df['id'].count())

print(df[
    (df['pathology'] == 1)
].count())
print(df[
    (df['pathology'] == 0)
].count())

# Hyperparameters
LR = 0.001 #Learning rate
BATCH_SIZE = 64
EPOCHS = 10

# Train/Validation splits
train_split = 0.75
valid_split = 1 - train_split

# Pytorch device to use. In this case using 
# Metal Performance Shaders implementation for Mac devices
device = torch.device('mps')

# Using the custom Torch dataset 
ds = CustomDS(df, db_path, 224, rotate=True)

# Creating data loaders 
train_data = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
valid_data = []

# Instantiate the custom CNN model
model = ZFNet()
model.to(device)

# Loss function and Optmizer initialization
loss_func = nn.BCELoss()
opt = torch.optim.Adam(params=model.parameters(), lr=LR)

print('Start training....')

## Training model
# Iterate over defined Epochs
for epoch in range(0, EPOCHS):
    model.train()

    correct = 0
    total = 0

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

        total += len(targets)
        correct += [int(round(predicts[i][0].item(), 0)) == targets[i][0].item() for i in range(0, len(predicts))].count(True)

    print('Epoch: {} | Loss: {:.3f} | Accuracy: {:.2f}%'.format(epoch + 1, loss.item(), correct / total * 100))

torch.save(model.state_dict(), os.path.join(cwd, 'model/model.torch'))
