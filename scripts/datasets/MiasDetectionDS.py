import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class MiasDetectionDS(Dataset):
    '''
        Definition of a custom Torch Dataset to be used on the detection.
    '''
    def __init__(self, file_path, num_win_x, num_win_y, step_x, step_y, model_img_size) -> None:
        super().__init__()

        self.file_path = file_path
        self.num_win_x = num_win_x
        self.num_win_y = num_win_y
        self.step_x = step_x
        self.step_y = step_y
        self.model_img_size = model_img_size
    
    def __len__(self):
        return self.num_win_x * self.num_win_y
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.toList()

        x = index % self.num_win_x
        y = index // self.num_win_x

        # Define crop bounds
        left = x * self.step_x
        top = y * self.step_y
        right = left + self.model_img_size
        bottom = top + self.model_img_size

        # Read image from db
        image = Image.open(self.file_path)
        # Crop Window
        image = image.crop((left, top, right, bottom))
        # Convert it to 0-1 float scale tensor
        image = T.ToTensor()(image)

        pos = [left + (self.model_img_size / 2), top + (self.model_img_size / 2)]

        return (image, pos)
