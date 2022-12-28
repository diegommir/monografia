import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler
import os
import numpy as np
import pandas as pd
import pydicom

# Definition of the Convolutional Neural Network
class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 7)
        # Result: [ Pc Px Py Pw Ph M/B M/C ]
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        return x

# Definition of a custom Torch Dataset based on the metadata
class CustomDS(Dataset):
    def __init__(self, metadata, db_path) -> None:
        super().__init__()

        self.metadata = metadata
        self.db_path = db_path
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.toList()


        row = self.metadata.iloc[index]

        # Get the full path to the image
        img_path = os.path.join(self.db_path, row[-1])
        image = pydicom.dcmread(img_path).pixel_array
        image = image.astype(np.int32)

        result = {
            'image': image,
            # Result: [ Pc Px Py Pw Ph M/B M/C ]
            # Cols: [ Pc, center width, center height, width, height, pathology, type ]
            'y': [1, row[5] / 2, row[6] / 2, row[5], row[6], row[2], row[7]]
        }
        return result

# Loading metadata
cwd = os.getcwd()
df = pd.read_csv(os.path.join(cwd, 'csv/metadb_train.csv'))

## Preping metadata
# Setting 1 to Malignant and 0 to Benign
df['pathology'] = df['pathology'].apply(lambda p: 1 if p == 'MALIGNANT' else 0)
# Setting 1 to Calc and 0 to Mass
df['type'] = df['type'].apply(lambda p: 1 if p == 'Calc' else 2)

#Maximum size allowed for the ROI images
img_cap = (512, 512)

print(df.head())
print('Total:', df['id'].count())
print('Bigger than {}x{}:'.format(img_cap[0], img_cap[1]), df[
    (df['width'] > img_cap[0]) |
    (df['height'] > img_cap[1])
]['id'].count())

#Filter images that are 600x600 or less
df = df[
    (df['width'] <= img_cap[0]) &
    (df['height'] <= img_cap[1])
]
print('Total filtered:', df['id'].count())

# Hyperparameters
LR = 1e-3 #Learning rate
BATCH_SIZE = 64
EPOCHS = 10

# Train/Validation splits
train_split = 0.75
valid_split = 1 - train_split

# Pytorch device to use. In this case using 
# Metal Performance Shaders implementation for Mac devices
device = torch.device('mps')

# Using the custom Torch dataset 
db_path = '/Volumes/Seagate/monografia/database/roi/train/'
ds = CustomDS(df, db_path)

temp = ds.__getitem__(10)
print(temp['y'])
'''
dataLoader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=RandomSampler(ds))

for i, sample in enumerate(dataLoader):
    print(i, sample['image'][0])
    break
'''