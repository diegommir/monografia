import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class ValidDS(Dataset):
    '''
        Definition of a custom Torch Dataset based on the metadata.
    '''
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
        img_path = os.path.join(self.db_path, '{}.bmp'.format(row['ref']))
        # Read image from db and convert to Tensor
        image = Image.open(img_path)
        # Convert it to 0-1 float scale tensor
        image = T.ToTensor()(image)

        target = [row['class_C'], row['class_M']]
        # Has to be float32 to work on MPS device
        target = torch.tensor(target, dtype=torch.float32)

        return image, target
