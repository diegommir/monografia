import torch
from torch import nn

class LeNet(nn.Module):
    '''
        Definition of a simple Convolutional Neural Network based on 
        LeNet network.
    '''

    def __init__(self) -> None:
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(61504, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 2),
            nn.Sigmoid()
        )

        # Result: [ Pc Px Py Pw Ph M/B M/C ]
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.fc(x)

        return x
