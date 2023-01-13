import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchmetrics.classification import MultilabelAccuracy, MultilabelConfusionMatrix
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

# Filter only the test images
df = df[df['db'] == 'test']
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
BATCH_SIZE = 64

# Pytorch device to use. In this case using 
# Metal Performance Shaders implementation for Mac devices
device = torch.device('mps')

# Using the custom Torch dataset 
ds = MiasDS(df, db_path)
test_data = DataLoader(ds, batch_size=BATCH_SIZE)

# Instantiate the custom CNN model
model = LeNet()
model.to(device)
model.load_state_dict(torch.load(os.path.join(cwd, 'model/model.torch')))

print('Testing model....')

## Testing model
metric = MultilabelConfusionMatrix(3, threshold=0.5)
accu = MultilabelAccuracy(3, threshold=0.5)

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

#TODO error!
#print(metric(preds, targs))
print('Accuracy: {:.2%}'.format(accu(preds, targs)))
