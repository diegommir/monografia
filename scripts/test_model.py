import os
import torch
from torch.utils.data import DataLoader
import pandas as pd

from models.LeNet import LeNet
from models.AlexNet import AlexNet
from datasets.CustomDS import CustomDS
import numpy as np

cwd = os.getcwd()
df = pd.read_csv(os.path.join(cwd, 'csv/metadb_test.csv'))
db_path = '/Volumes/Seagate/monografia/database/roi/test/'

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

#Filter images that are 600x600 or less
df = df[
    (df['width'] <= IMG_SIZE[0]) &
    (df['height'] <= IMG_SIZE[1])
]
print('Total filtered:', df['id'].count())

# Hyperparameters
BATCH_SIZE = 128

# Pytorch device to use. In this case using 
# Metal Performance Shaders implementation for Mac devices
device = torch.device('mps')

# Using the custom Torch dataset 
ds = CustomDS(df, db_path, 256)

# Creating data loaders 
test_data = DataLoader(ds, batch_size=BATCH_SIZE)

model = AlexNet()
model.to(device)
model.load_state_dict(torch.load(os.path.join(cwd, 'model/model.torch')))
model.eval()

print('Testing model...')

## Testing model
with torch.no_grad():
    correct = 0
    total = 0
    for (images, targets) in test_data:
        # Send tensors to the device
        images = images.to(device)
        targets = targets.to(device)

        predicts = model(images)

        total += len(targets)
        correct += [int(round(predicts[i][0].item(), 0)) == targets[i][0].item() for i in range(0, len(predicts))].count(True)
        print(total, correct)

    print('Accuracy: {:.2f}%'.format(correct / total * 100))
