import torch
from torch import nn

class ResNetCount(nn.Module):
    def __init__(self, pretrained_model):
        super(ResNetCount, self).__init__()
        self.pretrained_model = pretrained_model
        self.pretrained_model = nn.Sequential(*(list(self.pretrained_model.children())[:-1]))
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.Mish()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Mish()
        )
        self.classifier = nn.Linear(512, 1)
    
    def forward(self, x):
        x = self.pretrained_model(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.classifier(x)
        return x
