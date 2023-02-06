import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchmetrics.classification import MultilabelAccuracy, MultilabelRecall, MultilabelPrecision, MultilabelF1Score
import os
import pandas as pd

from models.LeNet import LeNet
from models.Custom import Custom
from models.AlexNet import AlexNet
from models.ZFNet import ZFNet
from datasets.ValidDS import ValidDS

# Loading metadata
cwd = os.getcwd()
df = pd.read_csv(os.path.join(cwd, 'csv/valid.csv'))
db_path = os.path.join(cwd, 'valid_db/roi')

# Generate the dummies
df_dummies = pd.get_dummies(df['class'], prefix='class')
df = df_dummies.join(df['ref'])
print(df.head())

# Hyperparameters
LR = 0.0005 #Learning rate
BATCH_SIZE = 64
EPOCHS = 100

# Pytorch device to use. In this case using 
# Metal Performance Shaders implementation for Mac devices
device = torch.device('mps')

# Train/Validation splits
train_split = 0.6
test_valid_split = 1 - train_split

# Using the custom Torch dataset 
ds = ValidDS(df, db_path)
(ds_train, ds_testvalid) = random_split(ds, [train_split, test_valid_split], generator=torch.Generator().manual_seed(42))
(ds_valid, ds_test) = random_split(ds, [.5, .5], generator=torch.Generator().manual_seed(42))

# Creating data loaders 
train_data = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
valid_data = DataLoader(ds_valid, batch_size=BATCH_SIZE)
test_data = DataLoader(ds_test, batch_size=BATCH_SIZE)

# Instantiate the custom CNN model
model = LeNet(14400)
model.to(device)

# Loss function and Optmizer initialization
loss_func = nn.CrossEntropyLoss()
opt = torch.optim.Adam(params=model.parameters(), lr=LR)

print('Start training....')

## Training model
num_labels = 2
accuracy = MultilabelAccuracy(num_labels)
recall = MultilabelRecall(num_labels)
precision = MultilabelPrecision(num_labels)
f1 = MultilabelF1Score(num_labels)

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

    print('Epoch: {} | Loss: {:.3f} | Accuracy: {:.2%} | Recall: {:.2%} | Precision: {:.2%} | F1: {:.2%}'
        .format(epoch + 1, loss.item(), accuracy(preds, targs), recall(preds, targs), precision(preds, targs), f1(preds, targs)))

torch.save(model.state_dict(), os.path.join(cwd, 'model/model.torch'))


print('')
print('')
print('')
print('Testing model....')
print('')

# Instantiate the custom CNN model
model = LeNet(14400)
model.to(device)
model.load_state_dict(torch.load(os.path.join(cwd, 'model/model.torch')))

# Testing model
preds = torch.tensor([])
targs = torch.tensor([])
with torch.no_grad():
    model.eval()

    # Iterate over validation dataset
    for (images, targets) in test_data:
        # Send tensors to the device
        images = images.to(device)
        targets = targets.to(device, dtype=torch.float32)

        # Predict classes
        predicts = model(images)

        # Calculate accuracy
        preds = torch.cat((preds, predicts.to('cpu')), dim=0)
        targs = torch.cat((targs, targets.to('cpu')), dim=0)

print('Accuracy: {:.2%} | Recall: {:.2%} | Precision: {:.2%} | F1: {:.2%}'
    .format(accuracy(preds, targs), recall(preds, targs), precision(preds, targs), f1(preds, targs)))
