import torch
from torch import nn

class Custom(nn.Module):
    '''
        Definition of a simple Convolutional Neural Network based on 
        LeNet network.
    '''

    def __init__(self, flatten_value) -> None:
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_value, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.fc(x)

        return x
