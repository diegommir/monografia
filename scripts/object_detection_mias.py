import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import MultilabelAccuracy
import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from models.LeNet import LeNet
from models.Custom import Custom
from models.AlexNet import AlexNet
from models.ZFNet import ZFNet
from datasets.MiasDetectionDS import MiasDetectionDS

# Loading metadata
cwd = os.getcwd()
df = pd.read_csv(os.path.join(cwd, 'csv/mias.csv'))
db_path = '/Volumes/Seagate/monografia/all-mias'
result_path = '/Volumes/Seagate/monografia/all-mias/results'
model_img_size = 64

# Pytorch device to use. In this case using 
# Metal Performance Shaders implementation for Mac devices
device = torch.device('mps')

# Instantiate the custom CNN model
model = Custom(12544)
model.to(device)
model.load_state_dict(torch.load(os.path.join(cwd, 'model/model.torch')))

def detect(row):
    file_name = '{}.pgm'.format(row['ref'])
    file_path = os.path.join(db_path, file_name)
    img = Image.open(file_path)

    num_win_x = int(img.size[0] / model_img_size * 2 - 1)
    num_win_y = int(img.size[1] / model_img_size * 2 - 1)
    step_x = model_img_size / 2
    step_y = model_img_size / 2

    print('Detecting for {}'.format(file_name))

    # Sliding window execution
    ds = MiasDetectionDS(file_path, num_win_x, num_win_y, step_x, step_y, model_img_size)
    test_data = DataLoader(ds, batch_size=64)

    # Classify
    with torch.no_grad():
        model.eval()

        for (images, positions) in test_data:
            images = images.to(device)

            predicts = model(images)
    
            for i in range(len(positions[0])):
                pos = (positions[0][i].item(), positions[1][i].item())
                pred = predicts[i].argmax().item()
                plt.text(pos[0], pos[1], pred, fontsize=8, color='red')

    plt.imshow(img, cmap='gray')
    plt.show()

df = df.sample(1)
df.apply(detect, axis=1)
