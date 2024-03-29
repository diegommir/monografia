import torch
from torch import nn

class ZFNet(nn.Module):
    '''
        Definition of a Convolutional Neural Network based on 
        ZFNet network.
    '''

    def __init__(self, flatten_value) -> None:
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 96, 7, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, 3, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_value, 4096),
            nn.ReLU(),
            nn.Linear(4096, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.fc(x)

        return x
