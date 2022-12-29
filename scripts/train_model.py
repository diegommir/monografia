import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler
import torchvision.transforms as T
import os
import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt

# Definition of the Convolutional Neural Network
class CustomCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, bias=False)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 6, 5)
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
        image = torch.from_numpy(image)
        # TODO: add a dimension to the image here
        print(image.shape)
        print(image)

        # Resize image
        width = row[5]
        height = row[6]
        pad_width = img_cap[0] - width
        pad_height = img_cap[1] - height
        image = T.Pad((0, 0, pad_width, pad_height))(image)

        # Target: [ Pc Px Py Pw Ph M/B M/C ]
        # Cols: [ Pc, center width, center height, width, height, pathology, type ]
        target = [1, int(row[5] / 2), int(row[6] / 2), row[5], row[6], row[7], row[2]]
        target = torch.tensor(target)

        return image, target

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
LR = 0.001 #Learning rate
BATCH_SIZE = 2
EPOCHS = 1

# Train/Validation splits
train_split = 0.75
valid_split = 1 - train_split

# Pytorch device to use. In this case using 
# Metal Performance Shaders implementation for Mac devices
device = torch.device('mps')

# Using the custom Torch dataset 
db_path = '/Volumes/Seagate/monografia/database/roi/train/'
ds = CustomDS(df, db_path)

# Instantiate the custom CNN model
model = CustomCNN()
model.to(device)

# Loss function and Optmizer initialization
loss_func = nn.CrossEntropyLoss()
opt = torch.optim.Adam(params=model.parameters(), lr=LR)

# Creating data loaders 
train_data = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
valid_data = []
test_data = []

# Print some images for validation of data
for i, (img, target) in enumerate(train_data):
    print('{} x {}'.format(len(img[0][0]), len(img[0])))
    print(img.size())
    im = plt.imshow(img.permute(1, 2, 0), cmap='gray')
    plt.show()
    if i == 0:
        break
'''

## Training model
# Iterate over defined Epochs
for epoch in range(0, EPOCHS):
    model.train()

    # Iterate over training dataset
    for (images, targets) in train_data:
        # Send tensors to the device
        images = images.to(device)
        targets = targets.to(device)

        # Forward propagation
        predicts = model(images)
        loss = loss_func(predicts, targets)

        # Backward propagation and optimization
        opt.zero_grad()
        loss.backward()
        opt.step()

    print('Epoch: {} | Loss: {:.3f}'.format(epoch + 1, loss.item()))

## Testing model
with torch.no_grad():
    model.eval()

    correct = 0
    total = 0
    for (images, targets) in test_data:
        # Send tensors to the device
        images = images.to(device)
        targets = targets.to(device)

        result = model(images)
        _, predicts = torch.max(predicts, 1)

        total = targets.size(0)
        correct += (predicts == targets).sum().item()

    print('Accuracy: {}%'.format(correct / total * 100))

'''