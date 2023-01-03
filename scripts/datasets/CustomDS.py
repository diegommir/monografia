import os
import torch
from torch.utils.data import Dataset
import pydicom
from PIL import Image
import numpy as np
import torchvision.transforms as T

class CustomDS(Dataset):
    '''
        Definition of a custom Torch Dataset based on the metadata.

        output_size: Int value representing the width and height of the output image
    '''
    def __init__(self, metadata, db_path, output_size, rotate=False) -> None:
        super().__init__()

        self.metadata = metadata
        self.db_path = db_path
        self.output_size = output_size
        self.rotate = rotate
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.toList()

        row = self.metadata.iloc[index]

        # Get the full path to the image
        img_path = os.path.join(self.db_path, row[-1])
        # Read image from db and convert to Tensor
        image = pydicom.dcmread(img_path).pixel_array
        # Scale down from 65k to 256 8bit
        image = np.uint8(image / 256)
        # Create a pillow image
        image = Image.fromarray(image)
        # Convert it to 0-1 float scale tensor
        image = T.ToTensor()(image)

        # Resize image
        width = row[5]
        height = row[6]
        pad_width = 0
        pad_height = 0
        # If the image fit the output, make it square with the output.
        # Will not resize after all.
        if width <= self.output_size and height <= self.output_size:
            pad_width = self.output_size - width
            pad_height = self.output_size - height
        # If the image is bigger than the output and is wider than longer, 
        # than make it square with the width
        elif width > height:
            pad_height = width - height
        # If the image is bigger than the output and is longer than wider, 
        # than make it square with the height
        else:
            pad_width = height - width

        # Make image square
        #image = T.Pad((0, 0, pad_width, pad_height))(image)
        # Resize it to the desired output size
        image = T.Resize(size=(self.output_size, self.output_size))(image)
        # Rotate images randomically
        if self.rotate:
            image = T.RandomHorizontalFlip()(image)
            image = T.RandomVerticalFlip()(image)

        # Target: [ Pc Px Py Pw Ph M/B M/C ]
        # Cols: [ Pc, center width, center height, width, height, pathology, type ]
        #target = [1, int(row[5] / 2), int(row[6] / 2), row[5], row[6], row[7], row[2]]
        target = [row[2] * 1.]
        # Has to be float32 to work on MPS device
        target = torch.tensor(target, dtype=torch.float32)

        return image, target
